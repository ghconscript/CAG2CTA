import os
import random
import numpy as np

# ================= 🔧 核心修复：解决 NumPy 兼容性报错 =================
# 必须在 import tigre 之前执行这两行
if not hasattr(np, 'int'):
    np.int = int  # 强行补回新版 NumPy 删除的 np.int
# =================================================================

import matplotlib.pyplot as plt
import nibabel as nib
import tigre
import tigre.algorithms as algs
from scipy.ndimage import zoom
from tqdm import tqdm

# ================= 1. 路径自动配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'src', 'datasets')

raw_nii_path = os.path.join(dataset_root, 'raw_nii')
save_path_GT = os.path.join(dataset_root, 'CCTA_GT')
save_path_BP = os.path.join(dataset_root, 'CCTA_BP')

os.makedirs(save_path_GT, exist_ok=True)
os.makedirs(save_path_BP, exist_ok=True)


# ================= 2. 核心逻辑 A：CCTA_split (提取 GT) =================
def process_gt_logic(fid):
    """完全保留你 datasimulation.py 中的逻辑"""
    file_path = os.path.join(raw_nii_path, f"{fid}.label.nii.gz")
    if not os.path.exists(file_path): return False, "找不到文件"

    img_nifti = nib.load(file_path)
    voxels_space = img_nifti.header['pixdim'][1:4]
    img = img_nifti.get_fdata()
    data = np.array(img)

    # 缩放逻辑
    data = zoom(data, (voxels_space[0], voxels_space[1], voxels_space[2]), order=0, mode='nearest') > 0
    pos = np.where(data > 0.5)
    xyzs = [pos[0], pos[1], pos[2]]

    x_diff = np.max(xyzs[0]) - np.min(xyzs[0])
    y_diff = np.max(xyzs[1]) - np.min(xyzs[1])
    z_diff = np.max(xyzs[2]) - np.min(xyzs[2])

    if x_diff < 128 and y_diff < 128 and z_diff < 128:
        x_gap, y_gap, z_gap = 128 - (x_diff + 1), 128 - (y_diff + 1), 128 - (z_diff + 1)
        xyzs[0] = xyzs[0] - np.min(xyzs[0]) + int(x_gap / 2)
        xyzs[1] = xyzs[1] - np.min(xyzs[1]) + int(y_gap / 2)
        xyzs[2] = xyzs[2] - np.min(xyzs[2]) + int(z_gap / 2)

        final_data = np.zeros((128, 128, 128), dtype=np.float32)
        final_data[xyzs[0], xyzs[1], xyzs[2]] = 1

        np.save(os.path.join(save_path_GT, f"{fid}.npy"), final_data)
        return True, final_data
    else:
        return False, f"尺寸超限 ({x_diff},{y_diff},{z_diff})"


# ================= 3. 核心逻辑 B：TIGRE 仿真 (提取 BP) =================
def process_bp_logic(fid, phantom):
    """完全保留你 datasimulation.py 中的仿真逻辑"""
    geo = tigre.geometry()
    geo.offDetector = np.array([0, 0])
    geo.accuracy = 1
    geo.COR = 0
    geo.rotDetector = np.array([0, 0, 0])
    geo.mode = "cone"

    # 第一次投影
    geo.nDetector = np.array([512, 512])
    d_spacing = 0.2779 + 0.001 * np.random.rand()
    geo.dDetector = np.array([d_spacing, d_spacing])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([128, 128, 128])
    v_size = 90 + 15 * np.random.rand()
    geo.sVoxel = np.array([v_size, v_size, v_size])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.DSD = 990 + 20 * np.random.rand() * random.choice((-1, 1))
    geo.DSO = 765 + 20 * np.random.rand() * random.choice((-1, 1))
    geo.offOrigin = np.array([0, 0, 0])

    angles = (np.array([[30 + 12 * np.random.rand() * random.choice((-1, 1)),
                         0 + 8 * np.random.rand() * random.choice((-1, 1)), 0]]) / 180 * np.pi)

    proj1 = tigre.Ax(phantom.copy(), geo, angles)
    # SIRT 迭代重建
    imgSIRT1 = algs.sirt(proj1 > 0, geo, angles, 1) > 0

    # 第二次投影
    geo.DSD = 1060 + 10 * np.random.rand() * random.choice((-1, 1))
    geo.offOrigin = np.array(
        [8 * np.random.rand() * random.choice((-1, 1)), 8 * np.random.rand() * random.choice((-1, 1)), 0])

    angles2 = (np.array([[0 + 8 * np.random.rand() * random.choice((-1, 1)),
                          30 + 12 * np.random.rand() * random.choice((-1, 1)), 0]]) / 180 * np.pi)

    proj2 = tigre.Ax(phantom.copy(), geo, angles2)

    geo.offOrigin = np.array([0, 0, 0])
    imgSIRT2 = algs.sirt(proj2 > 0, geo, angles2, 1) > 0

    recon = imgSIRT1.astype(np.int8) + imgSIRT2.astype(np.int8)
    np.save(os.path.join(save_path_BP, f"recon_{fid}.npy"), recon.astype(np.int8))
    return True


# ================= 4. 主程序 =================
if __name__ == '__main__':
    list_path = os.path.join(dataset_root, 'train.txt')
    if not os.path.exists(list_path):
        print("❌ 未找到 train.txt，请先运行 generate_split.py")
        exit()

    with open(list_path, 'r') as f:
        all_ids = [line.strip() for line in f.readlines() if line.strip()]

    #  试
    ids_to_process = all_ids
    #ids_to_process = all_ids[:1]

    print(f"🚀 正在基于原始逻辑处理 {len(ids_to_process)} 个样本...")

    for fid in tqdm(ids_to_process):
        ok_gt, result = process_gt_logic(fid)
        if ok_gt:
            process_bp_logic(fid, result)
        else:
            print(f"⚠️ ID {fid} 失败: {result}")

    print(f"\n🎉 处理完毕！成功生成数据。")