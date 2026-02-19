"""工具函数集合，比如指标计算（Dice 系数、IoU）、图像可视化、日志保存等"""
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.ndimage import binary_dilation

def dual_threshold_iteration(high_mask, low_mask):
    refined = high_mask.astype(bool)

    while True:
        dilated = binary_dilation(refined)
        new_refined = np.logical_and(dilated, low_mask)

        if np.array_equal(new_refined, refined):
            break

        refined = new_refined

    return refined.astype(np.uint8)

def dice_coeff(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
def get_metrics(predict, target, threshold=0.3):
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    predict_b = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()
    dice = dice_coeff(predict_b, target)
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    # auc = roc_auc_score(target, predict_b)
    if len(np.unique(target)) <= 1:
        auc = np.nan  # 如果只有一个类别，则将 ROC AUC 分数设置为 NaN
    else:
        auc = roc_auc_score(target, predict_b)
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    # sen = tp / (tp + fn)
    # 处理分母为零的情况
    if tp + fn == 0:
        sen = np.nan  # 或者将敏感度设置为其他特殊值
    else:
        sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    if pre + sen == 0:
        f1 = np.nan
    else:
        f1 = 2 * pre * sen / (pre + sen)
    return {
        "Dice": np.round(dice, 4),
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }

def plot_train(train_losses, val_losses, metrics_sums, save_path='training_metrics.png'):
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 提取各个指标的值
    dice_values = [metrics['Dice'] for metrics in metrics_sums]
    auc_values = [metrics['AUC'] for metrics in metrics_sums]
    f1_values = [metrics['F1'] for metrics in metrics_sums]
    acc_values = [metrics['Acc'] for metrics in metrics_sums]
    sen_values = [metrics['Sen'] for metrics in metrics_sums]
    spe_values = [metrics['Spe'] for metrics in metrics_sums]
    pre_values = [metrics['pre'] for metrics in metrics_sums]
    iou_values = [metrics['IOU'] for metrics in metrics_sums]
    # 绘制验证准确率和Dice系数曲线
    plt.subplot(1, 2, 2)
    plt.plot(dice_values, label='Dice')
    plt.plot(auc_values, label='AUC')
    plt.plot(f1_values, label='F1')
    plt.plot(acc_values, label='Acc')
    plt.plot(sen_values, label='Sen')
    plt.plot(spe_values, label='Spe')
    plt.plot(pre_values, label='Pre')
    plt.plot(iou_values, label='IOU')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Metrics over Epochs')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.neg_slope)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)