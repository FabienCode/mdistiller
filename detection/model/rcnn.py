# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from mdistiller.distillers.DKD import dkd_loss
from mdistiller.engine.kd_loss import mask_logits_loss
from mdistiller.engine.area_utils import AreaDetection, extract_regions, RegKD_pred

from .teacher import build_teacher
from .reviewkd import build_kd_trans, hcl

__all__ = ["RCNNKD", "ProposalNetwork"]


def rcnn_dkd_loss(stu_predictions, tea_predictions, gt_classes, alpha, beta, temperature, weight=1.0):
    stu_logits, stu_bbox_offsets = stu_predictions
    tea_logits, tea_bbox_offsets = tea_predictions
    gt_classes = torch.cat(tuple(gt_classes), 0).reshape(-1)
    loss_dkd = dkd_loss(stu_logits, tea_logits, gt_classes, alpha, beta, temperature)
    return {
        'loss_dkd': weight * loss_dkd,
    }


def dlm_rcnn_dkd_loss(stu_predictions, tea_predictions, gt_classes, alpha, beta, temperature):
    stu_logits, stu_bbox_offsets = stu_predictions
    tea_logits, tea_bbox_offsets = tea_predictions
    gt_classes = torch.cat(tuple(gt_classes), 0).reshape(-1)
    loss_dkd = dkd_loss(stu_logits, tea_logits, gt_classes, alpha, beta, temperature)
    return {
        'loss_dkd': loss_dkd,
    }


def aaloss(feature_student,
           feature_teacher,
           masks,
           scores):
    masks_stack = torch.stack(masks)
    # scores_expand = scores.unsqueeze(-1).unsqueeze(-1).expand_as(masks_stack)
    # weight_masks = masks_stack * scores_expand
    s_masks = masks_stack.sum(1)
    loss = F.mse_loss(feature_student * s_masks.unsqueeze(1), feature_teacher * s_masks.unsqueeze(1))
    return loss


def reg_logits_loss(stu_predictions, tea_predictions, temperature, mask=None):
    stu_logits, stu_bbox_offsets = stu_predictions
    tea_logits, tea_bbox_offsets = tea_predictions
    loss_reg_cls_kd = mask_kd_loss(stu_logits, tea_logits, temperature, mask=mask)
    return {
        'loss_reg_cls': loss_reg_cls_kd,
    }


@META_ARCH_REGISTRY.register()
class RCNNKD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            teacher_pixel_mean: Tuple[float],
            teacher_pixel_std: Tuple[float],
            teacher: nn.Module,
            kd_args,
            input_format: Optional[str] = None,
            teacher_input_format: Optional[str] = None,
            vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.teacher = teacher
        self.kd_args = kd_args
        # if self.kd_args.TYPE in ("ReviewKD", "ReviewDKD", "UniKD"):
        #     self.kd_trans = build_kd_trans(self.kd_args)
        # self.area_det = RegKD_pred(256, 256, 2, 81)
        # self.channel_mask = 0.95
        # self.conv_reg = ConvReg(256, 256)

        self.input_format = input_format
        self.teacher_input_format = teacher_input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("teacher_pixel_mean", torch.tensor(teacher_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("teacher_pixel_std", torch.tensor(teacher_pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # UniKD module
        self.feat2pro_s = featPro(256, 1024)
        self.feat2pro_t = featPro(256, 1024)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "kd_args": cfg.KD,
            "teacher": build_teacher(cfg),
            "teacher_input_format": cfg.TEACHER.INPUT.FORMAT,
            "teacher_pixel_mean": cfg.TEACHER.MODEL.PIXEL_MEAN,
            "teacher_pixel_std": cfg.TEACHER.MODEL.PIXEL_STD,

        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward_pure_roi_head(self, roi_head, features, proposals):
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_feature_full = box_features
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        return box_feature_full, box_features, predictions

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        losses = {}
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        sampled_proposals, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.kd_args.TYPE == "DKD":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            stu_predictions = self.forward_pure_roi_head(self.roi_heads, features, sampled_proposals)
            tea_predictions = self.forward_pure_roi_head(self.teacher.roi_heads, t_features, sampled_proposals)
            detector_losses.update(rcnn_dkd_loss(
                stu_predictions, tea_predictions, [x.gt_classes for x in sampled_proposals],
                self.kd_args.DKD.ALPHA, self.kd_args.DKD.BETA, self.kd_args.DKD.T))
        elif self.kd_args.TYPE == "ReviewKD":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            t_features = [t_features[f] for f in t_features]
            s_features = [features[f] for f in features]
            s_features = self.kd_trans(s_features)
            losses['loss_reviewkd'] = hcl(s_features, t_features) * self.kd_args.REVIEWKD.LOSS_WEIGHT
        # elif self.kd_args.TYPE == "ReviewDKD":
        #     teacher_images = self.teacher_preprocess_image(batched_inputs)
        #     t_features = self.teacher.backbone(teacher_images.tensor)
        #     # dkd loss
        #     box_full_feature_s, stu_predictions = self.forward_pure_roi_head(self.roi_heads, features,
        #                                                                      sampled_proposals)
        #     box_full_feature_t, tea_predictions = self.forward_pure_roi_head(self.teacher.roi_heads, t_features,
        #                                                                      sampled_proposals)
        #     detector_losses.update(rcnn_dkd_loss(
        #         stu_predictions, tea_predictions, [x.gt_classes for x in sampled_proposals],
        #         self.kd_args.DKD.ALPHA, self.kd_args.DKD.BETA, self.kd_args.DKD.T))
        #     # reviewkd loss
        #     t_features = [t_features[f] for f in t_features]
        #     s_features = [features[f] for f in features]
        #     s_features = self.kd_trans(s_features)
        #     losses['loss_reviewkd'] = hcl(s_features, t_features) * self.kd_args.REVIEWKD.LOSS_WEIGHT
        elif self.kd_args.TYPE == "RegKD":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            # logits loss
            stu_predictions = self.forward_pure_roi_head(self.roi_heads, features, sampled_proposals)
            tea_predictions = self.forward_pure_roi_head(self.teacher.roi_heads, t_features, sampled_proposals)
            t_features = [t_features[f] for f in t_features]
            s_features = [features[f] for f in features]
            f_s = self.conv_reg(s_features[-1])
            f_t = t_features[-1]
            heat_map, wh, offset, s_thresh, s_fc_mask = self.area_det(f_s, stu_predictions[0])
            t_heat_map, t_wh, t_offset, t_thresh, t_fc_mask = self.area_det(f_t, tea_predictions[0])
            tmp_mask = s_fc_mask - t_fc_mask
            fc_mask = torch.zeros_like(tmp_mask)
            fc_mask[tmp_mask == 0] = 1
            fc_mask[s_fc_mask == 0] = 0
            fc_mask[t_fc_mask == 0] = 0
            detector_losses.update(reg_logits_loss(stu_predictions, tea_predictions, 4, fc_mask.bool()))
            # Region Loss
            mask, scores = extract_regions(f_s, heat_map, wh, offset, 8, 3)
            loss_regkd = 2 * aaloss(f_s, t_features[-1], mask, scores)
            losses['loss_regkd'] = loss_regkd
            losses['loss_area'] = 1 * F.mse_loss(torch.cat((heat_map, wh, offset), dim=1),
                                                 torch.cat((t_heat_map, t_wh, t_offset,), dim=1))
            losses['loss_thresh'] = F.mse_loss(s_thresh, t_thresh)
        elif self.kd_args.TYPE == "UniKD":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            # dkd loss
            box_full_feature_s, gt_box_feature_s, stu_predictions = self.forward_pure_roi_head(self.roi_heads, features,
                                                                                               sampled_proposals)
            box_full_feature_t, gt_box_feature_t, tea_predictions = self.forward_pure_roi_head(self.teacher.roi_heads,
                                                                                               t_features,
                                                                                               sampled_proposals)
            detector_losses.update(rcnn_dkd_loss(
                stu_predictions, tea_predictions, [x.gt_classes for x in sampled_proposals],
                self.kd_args.DKD.ALPHA, self.kd_args.DKD.BETA, self.kd_args.DKD.T, 1.0))
            # reviewkd loss
            # t_features = box_full_feature_s
            # s_features = box_full_feature_t
            # s_features = self.kd_trans(s_features)
            # losses['loss_reviewkd'] = hcl(s_features, t_features) * self.kd_args.REVIEWKD.LOSS_WEIGHT
            # f_s = s_features[0]
            # f_t = t_features[0]
            s_cls, s_reg = self.feat2pro_s(box_full_feature_s)
            t_cls, t_reg = self.feat2pro_t(box_full_feature_t)
            losses['loss_feat2pro'] = 1.0 * (kd_loss(s_cls, t_cls, self.kd_args.DKD.T) + kd_loss(s_reg, t_reg, self.kd_args.DKD.T))
            # losses['loss_supp'] = 0.1 * (kd_loss(f_s_pro, gt_box_feature_s, self.kd_args.DKD.T) + kd_loss(f_t_pro, gt_box_feature_t, self.kd_args.DKD.T))
            # with torch.no_grad():
            #     supp_pre_s = self.roi_heads.box_predictor(f_s_pro)
            #     supp_pre_t = self.roi_heads.box_predictor(f_t_pro)
            # for i in range(2):
            #     loss_supp += 0.1 * (
            #             kd_loss(supp_pre_s[i], stu_predictions[i], self.kd_args.DKD.T) + kd_loss(supp_pre_t[i],
            #                                                                                      tea_predictions[i],
            #                                                                                      self.kd_args.DKD.T))
            loss_supp = 0.1 * (kd_loss(s_cls, stu_predictions[0], self.kd_args.DKD.T) + kd_loss(s_reg, tea_predictions[1], self.kd_args.DKD.T)) + \
                        0.1 * (kd_loss(t_cls, tea_predictions[0], self.kd_args.DKD.T) + kd_loss(t_reg, tea_predictions[1], self.kd_args.DKD.T))
            losses['loss_supp'] = loss_supp
        else:
            raise NotImplementedError(self.kd_args.TYPE)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return RCNNKD._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def teacher_preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.teacher_pixel_mean) / self.teacher_pixel_std for x in images]
        if self.input_format != self.teacher_input_format:
            images = [x.index_select(0, torch.LongTensor([2, 1, 0]).to(self.device)) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


def compute_fc_importance(fc_layer):
    importance = torch.norm(fc_layer.weight.data, p=1, dim=1)
    return importance


def prune_fc_layer(fc_layer, percentage):
    fc_importance = compute_fc_importance(fc_layer)
    _, sorted_index = torch.sort(fc_importance)
    num_prune = int(len(sorted_index) * percentage)
    prune_index = sorted_index[:num_prune]
    mask = torch.ones(len(sorted_index), dtype=torch.bool)
    mask[prune_index] = 0
    return mask


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_channel, t_channel, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        # s_N, s_C, s_H, s_W = s_shape
        # t_N, t_C, t_H, t_W = t_shape
        # if s_H == 2 * t_H:
        #     self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        # elif s_H * 2 == t_H:
        #     self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        # elif s_H >= t_H:
        #     self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        # else:
        #     raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.conv = nn.Conv2d(s_channel, t_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(t_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(t_channel, t_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(t_channel)
        self.conv_3 = nn.Conv2d(t_channel, t_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(t_channel)

    def forward(self, x):
        x = self.conv(x)
        # if self.use_relu:
        #     return self.relu(self.bn(x))
        # else:
        #     return self.bn(x)
        if self.use_relu:
            x = self.relu(self.bn(x))
        else:
            x = self.bn(x)
        x = self.conv_2(x)
        if self.use_relu:
            x = self.relu(self.bn2(x))
        else:
            x = self.bn2(x)
        x = self.conv_3(x)
        if self.use_relu:
            return self.relu(self.bn3(x))
        else:
            return self.bn3(x)


def mask_kd_loss(logits_student, logits_teacher, temperature, mask=None):
    if mask is not None:
        logits_student = logits_student * mask
        logits_teacher = logits_teacher * mask
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


class featPro(nn.Module):
    # Review KD version
    def __init__(self, in_channels, num_classes):
        super(featPro, self).__init__()
        # self.encoder = nn.Sequential(
        #     # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(in_channels),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
        #     # nn.BatchNorm2d(latent_dim),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.ReLU(inplace=True),
        # )
        self.fc_mu_cls = nn.Linear(in_channels, 81)
        self.fc_var_cls = nn.Linear(in_channels, 81)
        self.fc_mu_reg = nn.Linear(in_channels, 320)
        self.fc_var_reg = nn.Linear(in_channels, 320)
        # self.fc_mu = nn.Sequential(
        #     nn.Linear(in_channels, latent_dim),
        #     nn.Linear(latent_dim, num_classes)
        # )

    def encode_cls(self, x):
        # result = self.encoder(x)
        b, c, h, w = x.shape
        res_pooled = F.avg_pool2d(x, (h, w)).reshape(x.size(0), -1)
        # res_pooled = F.adaptive_avg_pool2d((h,w)).reshape(result.size(0), -1)
        # # result = result.view(result.size(0), -1)
        # res_pooled = self.avg_pool(result).reshape(result.size(0), -1)
        mu = self.fc_mu_cls(res_pooled)
        log_var = self.fc_var_cls(res_pooled)
        return mu, log_var

    def encode_reg(self, x):
        # result = self.encoder(x)
        b, c, h, w = x.shape
        res_pooled = F.avg_pool2d(x, (h, w)).reshape(x.size(0), -1)
        # res_pooled = F.adaptive_avg_pool2d((h,w)).reshape(result.size(0), -1)
        # # result = result.view(result.size(0), -1)
        # res_pooled = self.avg_pool(result).reshape(result.size(0), -1)
        mu = self.fc_mu_reg(res_pooled)
        log_var = self.fc_var_reg(res_pooled)
        return mu, log_var


    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu_cls, log_var_cls = self.encode_cls(x)
        mu_reg, log_var_reg = self.encode_reg(x)
        # z = torch.cat((mu, log_var), dim=1)
        # z = mu + log_var
        z_cls = self.reparameterize(mu_cls, log_var_cls)
        z_reg = self.reparameterize(mu_reg, log_var_reg)
        return z_cls, z_reg


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd
