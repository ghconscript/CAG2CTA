import torch
import numpy as np
import os
import sys
import numpy.core
import numpy.core.multiarray

# ==========================================
# 🌟 NumPy 2.x 降级到 1.x 的终极兼容魔法补丁
# 彻底解决 ModuleNotFoundError: No module named 'numpy._core'
# ==========================================
sys.modules['numpy._core'] = sys.modules['numpy.core']
sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']
sys.modules['numpy._core.umath'] = getattr(sys.modules['numpy.core'], 'umath', None)

from networks.generator import Generator
from dataRCA import Dataset, load_ids


def main():
    # 1. 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_results', 'checkpoints', 'best_checkpoint.tar')
    save_dir = os.path.join(current_dir, 'test_outputs')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到权重文件: {checkpoint_path}")
        return

    # 2. 初始化网络并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(in_channels=1, num_filters=64, class_num=1).to(device)

    print(f"🔄 正在加载历史最优权重...")

    # 放心加载，有了上面的魔法补丁，PyTorch 绝对不会再闹脾气了
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 智能识别你保存权重的字典 key
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'network' in checkpoint:
        model.load_state_dict(checkpoint['network'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ 权重加载成功！")

    # 3. 加载一个测试集数据
    test_ids = load_ids('test.txt')
    if not test_ids:
        print("❌ 测试集为空，请检查 test.txt！")
        return

    test_set = Dataset(test_ids)

    # 我们测试集里的第 9 个患者
    sample_idx = 9
    if sample_idx >= len(test_ids):
        sample_idx = 0  # 防止越界

    test_id = test_ids[sample_idx]
    print(f"🚀 正在测试患者 ID: {test_id} ...")

    # 获取数据并增加 Batch 维度: (C, H, W, D) -> (1, C, H, W, D)
    inputs, labels = test_set[sample_idx]
    inputs_tensor = inputs.unsqueeze(0).float().to(device)

    # 4. 模型推理
    with torch.no_grad():
        outputs = model(inputs_tensor)

    # 5. 保存结果为 numpy 数组 (去掉 batch 和 channel 维度，恢复成 H, W, D)
    output_np = outputs.squeeze().cpu().numpy()
    label_np = labels.squeeze().numpy()
    input_np = inputs.squeeze().numpy()

    # 存入 test_outputs 文件夹
    np.save(os.path.join(save_dir, f'pred_{test_id}.npy'), output_np)
    np.save(os.path.join(save_dir, f'gt_{test_id}.npy'), label_np)
    np.save(os.path.join(save_dir, f'input_{test_id}.npy'), input_np)

    print(f"🎉 测试完成！")
    print(f"文件已保存至: {save_dir}")
    print(f"👉 pred_{test_id}.npy (模型的预测结果)")
    print(f"👉 gt_{test_id}.npy (真实的冠脉金标准)")


if __name__ == '__main__':
    main()