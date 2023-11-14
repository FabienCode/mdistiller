import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

from mdistiller.models import cifar_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.cfg import CFG as cfg
from collections import OrderedDict

def cos_heat(stu, tea, val, name, layer=-1):
    _, stu_feat = fwd(stu, val, layer)
    _, tea_feat = fwd(tea, val, layer)
    selected_indices = np.random.choice(stu_feat.shape[0], size=50, replace=False)
    dkd_stu_feat_flatten = stu_feat.reshape(stu_feat.shape[0], -1)[selected_indices, :]
    dkd_tea_feat_flatten = tea_feat.reshape(tea_feat.shape[0], -1)[selected_indices, :]

    pca = PCA(n_components=2)
    dkd_stu_feat_flatten_pca = pca.fit_transform(dkd_stu_feat_flatten)
    dkd_tea_feat_flatten_pca = pca.fit_transform(dkd_tea_feat_flatten)

    cosine_similarity = np.dot(dkd_stu_feat_flatten_pca, dkd_tea_feat_flatten_pca.T) / (np.linalg.norm(dkd_stu_feat_flatten_pca, axis=1) * np.linalg.norm(dkd_tea_feat_flatten_pca, axis=1))
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    ratio = len(np.where(cosine_similarity > 0)[0])/cosine_similarity.size
    # print("{} positive ratio is {}".format(name, ratio))
    # plt.rcParams.update({'font.size': 14})
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(cosine_similarity, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    # plt.xlabel('Student Features')
    # plt.ylabel('Teacher Features')
    # plt.title('Cosine Similarity Heatmap between Teacher and Student Features')
    # png_name = name + ".png"
    # pdf_name = name + ".pdf"
    # plt.savefig(png_name, dpi=400)
    # plt.savefig(pdf_name, dpi=400)
    # # plt.show()
    # print("save {} successfully!".format(name))
    return cosine_similarity, ratio


def cos_heat(stu_feat, tea_feat, selected_indices):
    dkd_stu_feat_flatten = stu_feat.reshape(stu_feat.shape[0], -1)[selected_indices, :]
    dkd_tea_feat_flatten = tea_feat.reshape(tea_feat.shape[0], -1)[selected_indices, :]

    pca = PCA(n_components=2)
    dkd_stu_feat_flatten_pca = pca.fit_transform(dkd_stu_feat_flatten)
    dkd_tea_feat_flatten_pca = pca.fit_transform(dkd_tea_feat_flatten)

    # euclidean_similarity = np.sqrt(np.sum((dkd_stu_feat_flatten_pca - dkd_tea_feat_flatten_pca)**2, axis=1))
    cosine_similarity = np.dot(dkd_stu_feat_flatten_pca, dkd_tea_feat_flatten_pca.T) / (np.linalg.norm(dkd_stu_feat_flatten_pca, axis=1) * np.linalg.norm(dkd_tea_feat_flatten_pca, axis=1))
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    ratio = len(np.where(cosine_similarity > 0)[0])/cosine_similarity.size
    return cosine_similarity, ratio
    # return euclidean_similarity

def load_model(tea, stu, mpath):
    cfg.defrost()
    cfg.DISTILLER.STUDENT = stu
    cfg.DISTILLER.TEACHER = tea
    cfg.DATASET.TYPE = 'cifar100'
    cfg.freeze()
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    model = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
    fully_state = load_checkpoint(mpath)["model"]
    student_weights = OrderedDict()
    teacher_weights = OrderedDict()

    for key, value in fully_state.items():
        # 检查权重键是否包含 "student"
        if 'student' in key:
            key = key.replace("module.student.", "")
            student_weights[key] = value
        if 'teacher' in key:
            key = key.replace("module.teacher.", "")
            teacher_weights[key] = value
    # model.load_state_dict(load_checkpoint(mpath)["model"])
    # tea_model = cifar_model_dict[cfg.DISTILLER.TEACHER][0](num_classes=num_classes)
    # tea_model.load_state_dict(load_checkpoint(cifar_model_dict[cfg.DISTILLER.TEACHER][1])["model"])
    model.load_state_dict(student_weights)
    tea_model = cifar_model_dict[cfg.DISTILLER.TEACHER][0](num_classes=num_classes)
    tea_model.load_state_dict(teacher_weights)
    print("load {} successfully!".format(mpath))
    return model, tea_model, val_loader


def fwd(model, val_loader, layer, num_classes=100):
    model.eval()
    all_preds, all_feats = [], []
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            if i < 20:
                outputs, feats = model(data)
                preds = outputs
                all_preds.append(preds.data.cpu().numpy())
                all_feats.append(feats["feats"][layer].data.cpu().numpy())
            else:
                break

    all_preds = np.concatenate(all_preds, 0)
    all_feats = np.concatenate(all_feats, 0)
    return all_preds, all_feats



def save_heatmap(fit_sim, kd_sim, kr_sim, dkd_sim, unikd_sim, fit_name, kd_name, kr_name, dkd_name, uni_name):
    plot_sim(fit_sim, fit_name)
    plot_sim(kd_sim, kd_name)
    plot_sim(kr_sim, kr_name)
    plot_sim(dkd_sim, dkd_name)
    plot_sim(unikd_sim, uni_name)
    return

def plot_sim(sim, name):
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 10))
    sns.heatmap(sim, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xlabel('Student Features')
    plt.ylabel('Teacher Features')
    plt.title('Cosine Similarity Heatmap between Teacher and Student Features')
    png_name = name + ".png"
    pdf_name = name + ".pdf"
    plt.savefig(png_name, dpi=400)
    plt.savefig(pdf_name, dpi=400)
    # plt.show()
    print("save {} successfully!".format(name))





