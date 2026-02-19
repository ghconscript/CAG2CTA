import os
"测试 / 推理脚本，加载训练好的模型对新图像进行分割"
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from unet import Unet
from datasets import SimpleDatasets
from sklearn.metrics import roc_auc_score
"测试 / 推理脚本，加载训练好的模型对新图像进行分割"
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from unet import Unet
from datasets import SimpleDatasets
from sklearn.metrics import roc_auc_score
"测试 / 推理脚本，加载训练好的模型对新图像进行分割"
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from unet import Unet
from datasets import SimpleDatasets
from sklearn.metrics import roc_auc_score
from utils import dual_threshold_iteration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
res_dic_list = []
def dice_coeff(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
def get_metrics(predict, target, high_thr=0.80, low_thr=0.25):
    prob = torch.sigmoid(predict).cpu().detach().numpy()[0, 0]
    high_mask = prob > high_thr
    low_mask  = prob > low_thr
    predict_b = dual_threshold_iteration(high_mask, low_mask).flatten()

    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()

    dice = dice_coeff(predict_b, target)
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()

    auc = roc_auc_score(target, prob.flatten())
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-6)
    pre = tp / (tp + fp + 1e-6)
    sen = tp / (tp + fn + 1e-6)
    spe = tn / (tn + fp + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = 2 * pre * sen / (pre + sen + 1e-6)

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

def main():

    test_image_dir = "datasets/test_img"
    test_mask_dir = "datasets/test_mask"
    save_path = "run/unet"
    os.makedirs(save_path, exist_ok=True)
    test_dataset = SimpleDatasets(split='test', image_size=512)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True)
    # model = FR_UNet().to(device=DEVICE)
    model = Unet().to(device=DEVICE)
    model_path = os.path.join(save_path, "model.pth")  # 替换成你保存的模型文件路径
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        i = 0
        for x,y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            predict = model(x)
            res_dic = get_metrics(predict, y,high_thr=0.80, low_thr=0.25)
            res_dic_list.append(res_dic)
            prediction = torch.sigmoid(predict)
            # prediction = (prediction > 0.5).float()
            pre = prediction[0, 0, ...]
            prediction = torch.sigmoid(predict)[0, 0].cpu().detach().numpy()

            high_thr = 0.80
            low_thr = 0.25
            high_mask = prediction > high_thr
            low_mask = prediction > low_thr
            pred_dti = dual_threshold_iteration(high_mask, low_mask)

            gt = y[0, 0, ...].cpu().detach().numpy()

            cv2.imwrite(os.path.join(save_path, f"pre{i}.png"), np.uint8(pred_dti * 255))
            cv2.imwrite(os.path.join(save_path, f"gt{i}.png"), np.uint8(gt * 255))

            i = i + 1
    # 计算列表中每个指标的平均值
    avg_res_dic = {}
    for key in res_dic_list[0].keys():
        avg_res_dic[key] = np.round(np.mean([res_dic[key] for res_dic in res_dic_list]), 4)

    # 将平均值写入到文本文件中
    with open(os.path.join(save_path, "test_metrics.txt"), 'w') as f:
        f.write('Average Metrics:\n')
        print('Average Metrics:')
        for key, value in avg_res_dic.items():
            f.write(f'{key}: {value}\n')
            print(f'{key}: {value}')

if __name__ == '__main__':
    main()