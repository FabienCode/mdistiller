import multiprocessing
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import threading

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict


def run_program(cfg_file, log_wandb, resume, opts):
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(opts)
    # new_cfg = cfg.clone()  # 创建一个新的可修改的CfgNode对象
    # new_cfg.log_wandb = log_wandb.lower() == "true"
    # cfg.log_wandb = log_wandb.lower() == "true"
    cfg.freeze()
    main(cfg, resume, opts)


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.log_wandb:
        try:
            import wandb
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.log_wandb = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                    pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method('spawn')
#     configs = [  # 每个元素代表一个不同的配置
#         ("/data/home/cmshen/hym/project/new_mdistiller/mdistiller/configs/UniLogits/cifar/mlkd/res56_res20.yaml", "false", False,
#          ["SOLVER.BATCH_SIZE", "64", "Uni.LOSS.CE_WEIGHT", "1.0", "Uni.LOSS.LOGITS_WEIGHT", "1.0",
#           "Uni.LOSS.FEAT_KD_WEIGHT", "1.0", "Uni.LOSS.SUPP_WEIGHT", "0.1", "Uni.HINT_LAYER", "3", "Uni.SUPP_T", "4.0"]),
#         ("/data/home/cmshen/hym/project/new_mdistiller/mdistiller/configs/UniLogits/cifar/mlkd/res56_res20.yaml", "false", False,
#          ["SOLVER.BATCH_SIZE", "64", "Uni.LOSS.CE_WEIGHT", "2.0", "Uni.LOSS.LOGITS_WEIGHT", "1.0",
#           "Uni.LOSS.FEAT_KD_WEIGHT", "1.0", "Uni.LOSS.SUPP_WEIGHT", "0.1", "Uni.HINT_LAYER", "3", "Uni.SUPP_T", "4.0"]),
#         ("/data/home/cmshen/hym/project/new_mdistiller/mdistiller/configs/UniLogits/cifar/mlkd/res56_res20.yaml",
#          "false", False,
#          ["SOLVER.BATCH_SIZE", "64", "Uni.LOSS.CE_WEIGHT", "3.0", "Uni.LOSS.LOGITS_WEIGHT", "1.0",
#           "Uni.LOSS.FEAT_KD_WEIGHT", "1.0", "Uni.LOSS.SUPP_WEIGHT", "0.1", "Uni.HINT_LAYER", "3", "Uni.SUPP_T", "4.0"]),
#         ("/data/home/cmshen/hym/project/new_mdistiller/mdistiller/configs/UniLogits/cifar/mlkd/res56_res20.yaml",
#          "false", False,
#          ["SOLVER.BATCH_SIZE", "64", "Uni.LOSS.CE_WEIGHT", "4.0", "Uni.LOSS.LOGITS_WEIGHT", "1.0",
#           "Uni.LOSS.FEAT_KD_WEIGHT", "1.0", "Uni.LOSS.SUPP_WEIGHT", "0.1", "Uni.HINT_LAYER", "3", "Uni.SUPP_T", "4.0"]),
#         # ... 更多配置
#     ]
#
#     processes = []
#
#     for cfg_file, log_wandb, resume, opts in configs:
#         # 创建一个进程
#         p = multiprocessing.Process(target=run_program, args=(cfg_file, log_wandb, resume, opts))
#         p.start()
#         processes.append(p)
#
#     # 等待所有进程完成
#     for p in processes:
#         p.join()

if __name__ == "__main__":
    configs = [  # 每个元素代表一个不同的配置
        ("/data/home/cmshen/hym/project/new_mdistiller/mdistiller/configs/UniLogits/cifar/mlkd/res56_res20.yaml",
         "false", False,
         ["SOLVER.BATCH_SIZE", "64", "Uni.LOSS.CE_WEIGHT", "1.0", "Uni.LOSS.LOGITS_WEIGHT", "1.0",
          "Uni.LOSS.FEAT_KD_WEIGHT", "1.0", "Uni.LOSS.SUPP_WEIGHT", "0.1", "Uni.HINT_LAYER", "3", "Uni.SUPP_T", "4.0"]),
        ("/data/home/cmshen/hym/project/new_mdistiller/mdistiller/configs/UniLogits/cifar/mlkd/res56_res20.yaml",
         "false", False,
         ["SOLVER.BATCH_SIZE", "64", "Uni.LOSS.CE_WEIGHT", "2.0", "Uni.LOSS.LOGITS_WEIGHT", "1.0",
          "Uni.LOSS.FEAT_KD_WEIGHT", "1.0", "Uni.LOSS.SUPP_WEIGHT", "0.1", "Uni.HINT_LAYER", "3", "Uni.SUPP_T", "4.0"]),
        ("/data/home/cmshen/hym/project/new_mdistiller/mdistiller/configs/UniLogits/cifar/mlkd/res56_res20.yaml",
         "false", False,
         ["SOLVER.BATCH_SIZE", "64", "Uni.LOSS.CE_WEIGHT", "3.0", "Uni.LOSS.LOGITS_WEIGHT", "1.0",
          "Uni.LOSS.FEAT_KD_WEIGHT", "1.0", "Uni.LOSS.SUPP_WEIGHT", "0.1", "Uni.HINT_LAYER", "3", "Uni.SUPP_T", "4.0"]),
        ("/data/home/cmshen/hym/project/new_mdistiller/mdistiller/configs/UniLogits/cifar/mlkd/res56_res20.yaml",
         "false", False,
         ["SOLVER.BATCH_SIZE", "64", "Uni.LOSS.CE_WEIGHT", "4.0", "Uni.LOSS.LOGITS_WEIGHT", "1.0",
          "Uni.LOSS.FEAT_KD_WEIGHT", "1.0", "Uni.LOSS.SUPP_WEIGHT", "0.1", "Uni.HINT_LAYER", "3", "Uni.SUPP_T", "4.0"]),
        # ... 更多配置
    ]

    threads = []

    for cfg_file, log_wandb, resume, opts in configs:
        # 创建一个线程
        t = threading.Thread(target=run_program, args=(cfg_file, log_wandb, resume, opts))
        t.start()
        threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()
