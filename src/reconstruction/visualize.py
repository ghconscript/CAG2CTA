import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 🌟 修改这里：填入你刚才生成的真实文件名
# ==========================================
patient_id = '392'  # 比如 '265'，填入你刚才测试出来的那个 ID
input_file = f'test_outputs/input_{patient_id}.npy'
gt_file    = f'test_outputs/gt_{patient_id}.npy'
pred_file  = f'test_outputs/pred_{patient_id}.npy'

def main():
    if not os.path.exists(pred_file):
        print(f"❌ 找不到文件: {pred_file}，请检查 patient_id 是否填对！")
        return

    # 1. 加载 3D 矩阵数据
    print("🔄 正在加载 3D 数据...")
    inp = np.load(input_file)
    gt = np.load(gt_file)
    pred = np.load(pred_file)

    # 医疗影像通常包含极小的值，对于预测结果，我们可以加个简单的阈值二值化一下，看着更清晰
    # 如果预测出的概率大于 0.5，我们就认为它是血管
    pred_binary = (pred > 0.5).astype(np.float32)

    # 准备保存图片的文件夹
    os.makedirs('visual_results', exist_ok=True)

    # ==========================================
    # 视角一：单层切片对比 (像切西瓜一样看中间那一层)
    # ==========================================
    print("📸 正在生成【中心切片】对比图...")
    mid_z = inp.shape[2] // 2  # 取 Z 轴最中间的那一层

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(inp[:, :, mid_z], cmap='gray')
    axes[0].set_title('Input (Original)')
    axes[0].axis('off')

    axes[1].imshow(gt[:, :, mid_z], cmap='gray')
    axes[1].set_title('Ground Truth (Real Vessel)')
    axes[1].axis('off')

    axes[2].imshow(pred_binary[:, :, mid_z], cmap='gray')
    axes[2].set_title('Prediction (AI Generated)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'visual_results/slice_{patient_id}.png', dpi=300)
    plt.close()

    # ==========================================
    # 视角二：最大密度投影 MIP (看血管走势的绝杀技巧)
    # ==========================================
    print("📸 正在生成【最大密度投影 MIP】对比图...")
    # MIP 的原理：把 128 层切片叠在一起，每个像素点取这 128 层里的最大值，能完美展现 3D 血管的树状分支！
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.max(inp, axis=2), cmap='gray')
    axes[0].set_title('Input MIP')
    axes[0].axis('off')

    axes[1].imshow(np.max(gt, axis=2), cmap='gray')
    axes[1].set_title('Ground Truth MIP')
    axes[1].axis('off')

    axes[2].imshow(np.max(pred_binary, axis=2), cmap='gray')
    axes[2].set_title('Prediction MIP')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'visual_results/mip_{patient_id}.png', dpi=300)
    plt.close()

    print("🎉 可视化完成！")
    print("请在 VS Code 左侧的文件树中，打开 `visual_results` 文件夹查看生成的图片！")

if __name__ == '__main__':
    main()