import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn

from mdistiller.models import cifar_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.cfg import CFG as cfg
from collections import OrderedDict

# set a common max-value of the difference for fair comparsion between different methods
MAX_DIFF = 3.0

# visualize the difference between the teacher's output logits and the student's
def get_output_metric(model, val_loader, num_classes=100):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            outputs, _ = model(data)
            preds = outputs
            all_preds.append(preds.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    matrix = np.zeros((num_classes, num_classes))
    cnt = np.zeros((num_classes, 1))
    for p, l in zip(all_preds, all_labels):
        cnt[l, 0] += 1
        matrix[l] += p
    matrix /= cnt
    return matrix


def get_tea_stu_diff(tea, stu, mpath, max_diff):
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
    print("load model successfully!")
    ms = get_output_metric(model, val_loader)
    mt = get_output_metric(tea_model, val_loader)
    diff = np.abs((ms - mt)) / max_diff
    for i in range(100):
        diff[i, i] = 0
    print('max(diff):', diff.max())
    print('mean(diff):', diff.mean())
    fig = seaborn.heatmap(diff, vmin=0, vmax=1.0, cmap="PuBuGn")
    max_num = str(diff.max()).split('.')[0] + '.' + str(diff.max()).split('.')[1][:2]
    mean_num = str(diff.mean()).split('.')[0] + '.' + str(diff.mean()).split('.')[1][:2]
    fig_name = "max_diff: " + max_num + "mean_diff:" + mean_num
    path_name = "/home/fabien/Documents/project/2d/mdistiller/tools/visualizations/corrimg/"
    final_name = path_name + fig_name + "unikd_324_84_best.png"
    heatmap = fig.get_figure()
    heatmap.savefig(final_name, dpi = 400)
    # plt.show()

def get_tea_stu_cdf(tea, stu, mpath, max_diff):
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
    print("load model successfully!")

    # 获取输出矩阵
    stu_matrix = get_output_metric(model, val_loader)
    tea_matrix = get_output_metric(tea_model, val_loader)
    
    # 获取输出矩阵
    # stu_matrix = get_output_metric(model, val_loader, num_classes)
    # tea_matrix = get_output_metric(tea_model, val_loader, num_classes)
    
    # 计算输出差异并归一化
    diff_matrix = np.abs(stu_matrix - tea_matrix) / max_diff
    diff_matrix = diff_matrix[np.triu_indices_from(diff_matrix, k=1)]  # 取上三角矩阵的值，排除对角线
    
    # 计算CDF
    sorted_diff = np.sort(diff_matrix)
    cumulative = np.cumsum(sorted_diff) / np.sum(sorted_diff)

    return sorted_diff, cumulative

def main_logits(tea, stu, mpath, our_path, max_diff):
    # 绘制CDF图

    sorted_diff_fk, cumulative_fk = get_tea_stu_cdf(tea, stu, mpath, max_diff)
    sorted_diff_our, cumulative_our = get_tea_stu_cdf(tea, stu, our_path, max_diff)
    plt.figure(figsize=(8, 4))
    # 绘制第一组数据的CDF线
    plt.plot(sorted_diff_fk, cumulative_fk, label='FitNet & KD', color='blue')

    # 绘制第二组数据的CDF线
    plt.plot(sorted_diff_our, cumulative_our, label='UniKD', color='red')

    # 添加图例
    plt.legend()

    # 添加图表标题和轴标签
    plt.title('CDF Comparison between differnet methods')
    plt.xlabel('Difference')
    plt.ylabel('Cumulative Distribution')

    # 显示网格
    plt.grid(True)

    path_name = "/home/fabien/Documents/project/2d/mdistiller/tools/visualizations/corrimg/"
    final_name = path_name + "uni240_mlkd240_324_84.pdf"
    
    # 保存图像
    plt.savefig(final_name, dpi=300)
    # plt.savefig('plot.pdf')

if __name__ == '__main__':
    mpath = '/home/fabien/Documents/project/2d/mdistiller/output/final/dkd/mlkd_324_84/epoch_240'
    our_path = '/home/fabien/Documents/project/2d/mdistiller/output/final/UniKD/res324_84_77.71/epoch_240'
    # get_tea_stu_diff('resnet32x4', 'resnet8x4', mpath, 3.0)
    main_logits('resnet32x4', 'resnet8x4', mpath, our_path, 3.0)