import os.path
"训练脚本，定义训练循环、损失函数、优化器等"
import torch
import torch.nn as nn

from unet import Unet
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import get_metrics, plot_train
from datasets import SimpleDatasets
from torch.utils.data import DataLoader

LEARNING_RATE = 1e-4
weight_decay = 1e-5
BATCH_SIZE = 4
NUM_EPOCH = 100
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
"""内存锁定，将数据加载到内存固定区域
不加载预训练模型，从头训练"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# gpu_id = 2  # 请根据实际情况选择合适的 GPU 编号
# DEVICE = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
# TRAIN_IMG_DIR = "datasets/train_img"
# TRAIN_MASK_DIR = "datasets/train_mask"
VAL_IMG_DIR = "datasets/val_img"
VAL_MASK_DIR = "datasets/val_mask"
IMAGE_SIZE = 512
save_path = "run/unet"
os.makedirs(save_path, exist_ok=True)
"""负责执行单轮训练中的向前传播、损失计算、反向传播和参数更新"""
def train_fn(train_loader, model, loss_fn, optimizer, scheduler):
    loop = tqdm(train_loader)#提供进度可视化
    total_loss = 0.0#当前轮次的总损失
    for index, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()#梯度清零
        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()#反向传播计算梯度=
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    scheduler.step()
    return total_loss / len(train_loader)

def check_accuracy(loader, model, loss_fn, DEVICE="cuda"):
    model.eval()#切换到验证模式，关闭dropout和BatchNorm的更新
    metrics_sum = {"Dice": 0, "AUC": 0, "F1": 0, "Acc": 0, "Sen": 0, "Spe": 0, "pre": 0, "IOU": 0}
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            predictions = model(x)
            loss = loss_fn(predictions, y)
            metrics = get_metrics(predictions, y)
            for key in metrics_sum.keys():
                metrics_sum[key] += metrics[key]
        # 计算平均值
        for key in metrics_sum.keys():
            metrics_sum[key] = round(metrics_sum[key] / len(loader), 4)
    print("Metrics:", metrics_sum)
    model.train()
    return loss.item(), metrics_sum

def main():
    train_dataset = SimpleDatasets(split='train', image_size=512)
    val_dataset = SimpleDatasets(split='val', image_size=512)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    # model = FR_UNet().to(device=DEVICE)
    model = Unet().to(device=DEVICE)
    pos_weight = 1.0
    pos_weight = torch.tensor(pos_weight).cuda()
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCH)
    train_losses = []
    val_losses = []
    metrics_sums = []
    best_dice=0.0
    best_epoch=0
    for index in range(NUM_EPOCH):
        print("Current epoch:",index)
        train_loss = train_fn(train_loader, model, loss_fn, optimizer, scheduler)
        train_loss = round(train_loss, 4)
        train_losses.append(train_loss)
        val_loss, metrics_sum = check_accuracy(val_loader, model, loss_fn=loss_fn, DEVICE=DEVICE)
        val_loss = round(val_loss, 4)
        val_losses.append(val_loss)
        metrics_sums.append(metrics_sum)
        current_dice=metrics_sum["Dice"]
        if current_dice>best_dice:
            best_dice=current_dice
            best_epoch=index+1
            best_model_path=os.path.join(save_path,f"best_model_epoch_{best_epoch}_dice_{best_dice}.pth")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_loss': val_loss
            }, best_model_path)
            print(f"✅ 更新最优模型：第{best_epoch}轮，Dice={best_dice}，已保存至 {best_model_path}")
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    with open(os.path.join(save_path, 'best_model_info.txt'), 'w') as f:
        f.write(f"最优模型轮数：{best_epoch}\n")
        f.write(f"最优Dice值：{best_dice}\n")
        f.write(f"最优轮次验证损失：{val_losses[best_epoch - 1]}\n")
    plot_train(train_losses, val_losses, metrics_sums, save_path=os.path.join(save_path, 'training_metrics.png'))
    with open(os.path.join(save_path, 'train_metrics.txt'), 'w') as f:
        for index in range(len(train_losses)):
            train_loss = train_losses[index]
            val_loss = val_losses[index]
            metrics_sum = metrics_sums[index]
            f.write(f'Epoch{index+1}: Train Loss: {train_loss}, Val Loss: {val_loss}, Metrics Sum: {metrics_sum}\n')


if __name__ == '__main__':
    main()

