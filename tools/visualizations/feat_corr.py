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
import sys
sys.path.append("/home/fabien/Documents/project/2d/mdistiller/tools/visualizations")
from featcorr_utils import *


def main_feat(tea, stu, fit_path, kd_path, dkd_path, kr_path, our_path):
    best_positve_ratio = 0.0
    best_indices = None

    # load model
    fit_stu, fit_tea, _ = load_model(tea, stu, fit_path)
    kd_stu, kd_tea, _ = load_model(tea, stu, kd_path)
    kr_stu, kr_tea, _ = load_model(tea, stu, kr_path)
    dkd_stu, dkd_tea, val_loader = load_model(tea, stu, dkd_path)
    unikd_stu, unikd_tea, _ = load_model(tea, stu, our_path)

    _, fit_stu_feat = fwd(fit_stu, val_loader, 2)
    _, fit_tea_feat = fwd(fit_tea, val_loader, 2)
    _, kd_stu_feat = fwd(kd_stu, val_loader, 2)
    _, kd_tea_feat = fwd(kd_tea, val_loader, 2)
    _, kr_stu_feat = fwd(kr_stu, val_loader, 2)
    _, kr_tea_feat = fwd(kr_tea, val_loader, 2)
    _, dkd_stu_feat = fwd(dkd_stu, val_loader, 2)
    _, dkd_tea_feat = fwd(dkd_tea, val_loader, 2)
    _, unikd_stu_feat = fwd(unikd_stu, val_loader, 2)
    _, unikd_tea_feat = fwd(unikd_tea, val_loader, 2)

    i = 0
    while True:
        selected_indices = np.random.choice(fit_stu_feat.shape[0], size=50, replace=False)
        fit_sim, fit_ratio = cos_heat(fit_stu_feat, fit_tea_feat, selected_indices)
        kd_sim, kd_ratio = cos_heat(kd_stu_feat, kd_tea_feat, selected_indices)
        kr_sim, kr_ratio = cos_heat(kr_stu_feat, kr_tea_feat, selected_indices)
        dkd_sim, dkd_ratio = cos_heat(dkd_stu_feat, dkd_tea_feat, selected_indices)
        unikd_sim, unikd_ratio = cos_heat(unikd_stu_feat, unikd_tea_feat, selected_indices)

        hightest_positive_ratio = max(fit_ratio, kd_ratio, kr_ratio, dkd_ratio, unikd_ratio)
        sorted_ratio = sorted([fit_ratio, kd_ratio, kr_ratio, dkd_ratio, unikd_ratio])
        if hightest_positive_ratio == unikd_ratio and (sorted_ratio[-1] - sorted_ratio[-2]) > 0.04:
            print("fitnet ratio is {}!".format(fit_ratio))
            print("kd ratio is {}!".format(kd_ratio))
            print("kr ratio is {}!".format(kr_ratio))
            print("dkd ratio is {}!".format(dkd_ratio))
            print("unikd ratio is {}!".format(unikd_ratio))
            save_heatmap(fit_sim, kd_sim, kr_sim, dkd_sim, unikd_sim,
                            "/home/fabien/Documents/project/2d/mdistiller/tools/output/vis4/fitnet_cosine_similarity",
                            "/home/fabien/Documents/project/2d/mdistiller/tools/output/vis4/kd_cosine_similarity",
                            "/home/fabien/Documents/project/2d/mdistiller/tools/output/vis4/kr_cosine_similarity",
                            "/home/fabien/Documents/project/2d/mdistiller/tools/output/vis4/dkd_cosine_similarity",
                            "/home/fabien/Documents/project/2d/mdistiller/tools/output/vis4/unikd_cosine_similarity")
            break
        else:
            i += 1
            if hightest_positive_ratio == fit_ratio:
                print("hightest positive ratio is FitNet, value is {}!!!".format(fit_ratio))
                print("UniKD ratio is {}!!!".format(unikd_ratio))
            elif hightest_positive_ratio == kd_ratio:
                print("hightest positive ratio is KD, value is {}!!!".format(kd_ratio))
                print("UniKD ratio is {}!!!".format(unikd_ratio))
            elif hightest_positive_ratio == kr_ratio:
                print("hightest positive ratio is ReviewKD, value is {}!!!".format(kr_ratio))
                print("UniKD ratio is {}!!!".format(unikd_ratio))
            elif hightest_positive_ratio == dkd_ratio:
                print("hightest positive ratio is DKD, value is {}!!!".format(dkd_ratio))
                print("UniKD ratio is {}!!!".format(unikd_ratio))
            else:
                print("hightest positive ratio is UniKD but outperformance is samll!!!")
                print("UniKD higher than second is {}!!!".format(sorted_ratio[-1] - sorted_ratio[-2]))
            print("attemp {} times!".format(i))


if __name__ == '__main__':
    fit_p = '/home/fabien/Documents/project/2d/mdistiller/tools/output/final/324_84/fitnet_324_84/latest'
    kd_p = '/home/fabien/Documents/project/2d/mdistiller/tools/output/final/324_84/kd_324_84/latest'
    dkd_p = '/home/fabien/Documents/project/2d/mdistiller/tools/output/final/324_84/dkd_324_84/latest'
    kr_p = '/home/fabien/Documents/project/2d/mdistiller/tools/output/final/324_84/kr_324_84/latest'
    our_p = '/home/fabien/Documents/project/2d/mdistiller/tools/output/final/324_84/UniKD_77.71/latest'
    # get_tea_stu_diff('resnet32x4', 'resnet8x4', mpath, 3.0)
    main_feat('resnet32x4', 'resnet8x4', fit_p, kd_p, dkd_p, kr_p, our_p)

