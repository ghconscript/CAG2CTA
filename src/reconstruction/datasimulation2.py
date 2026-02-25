import os
import random
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tigre
import tigre.algorithms as algs
from scipy.ndimage import zoom

np.int = int # =====================================================================
# 模块一：NIfTI 医学图像分离模块 (ImageCAS .nii -> 3D Numpy)
# =====================================================================

def batch_process_imagecas():
    """
    外层发号施令函数：负责遍历 ImageCAS 文件夹，喂给处理中心
    """
    # 1. 在这里写死你的 ImageCAS 文件夹路径 (请替换成你的真实路径)
    input_dir = "../datasets/ImageCAS_labels/"

    # 2. 提前建好保存结果的文件夹
    os.makedirs("./split_one", exist_ok=True)
    os.makedirs("./split_two", exist_ok=True)

    # 3. 筛选出所有的 .nii 或 .nii.gz 文件
    if not os.path.exists(input_dir):
        print(f"❌ 找不到文件夹: {input_dir}")
        return

    all_files = os.listdir(input_dir)
    nii_files = [f for f in all_files if f.endswith('.nii') or f.endswith('.nii.gz')]
    print(f"🚀 总共找到 {len(nii_files)} 个数据，开始连通域分离处理...")

    # 4. 开启流水线
    for i, file_name in enumerate(nii_files):
        full_path = os.path.join(input_dir, file_name)
        save_name = str(i + 1)  # 保存为 1, 2, 3...

        print(f"[{i + 1}/{len(nii_files)}] 正在拆分: {file_name}")
        process_single_ccta(full_path, save_name)

    print("✅ 全部数据分离完成！\n")


def process_single_ccta(file_path, save_name):
    """
    内层干活函数：读取单一的 nii 文件，拆分成两个连通域并保存
    """
    img_nifti = nib.load(file_path)
    voxels_space = img_nifti.header['pixdim'][1:4]
    img = img_nifti.get_fdata()
    data = np.array(img)

    data = zoom(data, (voxels_space[0], voxels_space[1], voxels_space[2]), order=0, mode='nearest') > 0
    pos = np.where(data > 0.5)
    xyzs = [pos[0], pos[1], pos[2]]

    v_min, v_max = np.min(xyzs[0]), np.max(xyzs[0])
    xyzs[0] = xyzs[0] - v_min
    x_diff = v_max - v_min

    v_min, v_max = np.min(xyzs[1]), np.max(xyzs[1])
    xyzs[1] = xyzs[1] - v_min
    y_diff = v_max - v_min

    v_min, v_max = np.min(xyzs[2]), np.max(xyzs[2])
    xyzs[2] = xyzs[2] - v_min
    z_diff = v_max - v_min

    if x_diff < 128 and y_diff < 128 and z_diff < 128:
        x_gap = 128 - (x_diff + 1)
        y_gap = 128 - (y_diff + 1)
        z_gap = 128 - (z_diff + 1)

        xyzs[0] = xyzs[0] + int(x_gap / 2)
        xyzs[1] = xyzs[1] + int(y_gap / 2)
        xyzs[2] = xyzs[2] + int(z_gap / 2)

        data = np.zeros((128, 128, 128))
        data[xyzs[0], xyzs[1], xyzs[2]] = 1

        w, h, d = data.shape
        coords = []
        flag = False
        for i in range(w):
            if flag: break
            for j in range(h):
                if flag: break
                for k in range(d):
                    if data[i, j, k] > 0:
                        coords.append([i, j, k])
                        flag = True
                        break

        for [x, y, z] in coords:
            for cx in [x - 1, x, x + 1]:
                for cy in [y - 1, y, y + 1]:
                    for cz in [z - 1, z, z + 1]:
                        c_coord = [cx, cy, cz]
                        if not (c_coord in coords):
                            if cx > -1 and cx < w and cy > -1 and cy < h and cz > -1 and cz < d:
                                if data[cx, cy, cz] > 0:
                                    coords.append(c_coord)

        coords = np.transpose(np.array(coords))
        data[coords[0], coords[1], coords[2]] = 0
        np.save(f"./split_one/{save_name}", data.astype('int8'))

        data = data * 0
        data[coords[0], coords[1], coords[2]] = 1
        np.save(f"./split_two/{save_name}", data.astype('int8'))
    else:
        print(f'⚠️ 忽略文件 (尺寸超限): {file_path}')


# =====================================================================
# 模块二：物理投影与位移伪影模拟模块 (3D Numpy -> 2D X光片 + 错位 3D)
# =====================================================================

def batch_generate_projections():
    """
    外层发号施令函数：读取提取好的 3D 模型 (.npy)，进行流水线物理投影
    """
    # 1. 假设你想对 split_one 里分离出来的完美 3D 血管进行物理投影测试
    input_dir = './split_two/'  # 请根据你的需要修改数据来源文件夹

    # 2. 建好所有保存生成的假数据的文件夹
    os.makedirs('./CCTA_first_proj/', exist_ok=True)
    os.makedirs('./CCTA_second_proj/', exist_ok=True)
    os.makedirs('./CCTA_BP/', exist_ok=True)

    # 3. 初始化 TIGRE 基础几何环境 (放循环外面，提升效率)
    geo = tigre.geometry()
    geo.offDetector = np.array([0, 0])
    geo.accuracy = 1
    geo.COR = 0
    geo.rotDetector = np.array([0, 0, 0])
    geo.mode = "cone"

    # 4. 获取文件列表
    if not os.path.exists(input_dir):
        print(f"❌ 找不到文件夹: {input_dir}")
        return

    all_files = os.listdir(input_dir)
    npy_files = [f for f in all_files if f.endswith('.npy')]
    print(f"🚀 总共找到 {len(npy_files)} 个 3D 模型，开始生成 X光片 和 残次品 3D 矩阵...")

    # 5. 开启流水线
    for i, file_name in enumerate(npy_files):
        full_path = os.path.join(input_dir, file_name)
        save_name = file_name.replace('.npy', '')  # 去掉后缀，保留数字编号

        print(f"[{i + 1}/{len(npy_files)}] 正在物理模拟: {file_name}")
        generate_single_projection_RCA(full_path, save_name, geo)

    print("✅ 所有数据的物理投影和 Ill-posed 重建完成！")


def generate_single_projection_RCA(phantom_path, save_name, geo):
    """
    内层干活函数：对单个 3D 模型拍两张 X 光片，并故意制造心脏跳动带来的位移伪影
    """
    phantom = np.load(phantom_path).astype(np.float32)

    # --- 探测器和体素参数随机化 ---
    geo.nDetector = np.array([512, 512])
    d_spacing = 0.2779 + 0.001 * np.random.rand()
    geo.dDetector = np.array([d_spacing, d_spacing])
    geo.sDetector = geo.nDetector * geo.dDetector

    geo.nVoxel = np.array([128, 128, 128])
    v_size = 90 + 15 * np.random.rand()
    geo.sVoxel = np.array([v_size, v_size, v_size])
    geo.dVoxel = geo.sVoxel / geo.nVoxel

    # ==========================================
    # 拍摄第一张 X 光片 (正常状态)
    # ==========================================
    geo.DSD = 990 + 20 * np.random.rand() * random.choice((-1, 1))
    geo.DSO = 765 + 20 * np.random.rand() * random.choice((-1, 1))
    geo.offOrigin = np.array([0, 0, 0])

    angle_one_pri = 30 + 12 * np.random.rand() * random.choice((-1, 1))
    angle_one_sec = 0 + 8 * np.random.rand() * random.choice((-1, 1))
    angles_1 = np.array([[angle_one_pri, angle_one_sec, 0]]) / 180 * np.pi

    projections_1 = tigre.Ax(phantom.copy(), geo, angles_1) > 0

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.imshow(projections_1[0], cmap=plt.get_cmap('Greys'))
    plt.savefig(f'./CCTA_first_proj/{save_name}.png')
    plt.close(fig1)  # ⚠️ 极其重要：释放内存防崩溃

    imgSIRT_one = algs.sirt(projections_1, geo, angles_1, 1) > 0

    # ==========================================
    # 拍摄第二张 X 光片 (发生心脏跳动位移)
    # ==========================================
    geo.DSD = 1060 + 10 * np.random.rand() * random.choice((-1, 1))
    geo.DSO = geo.DSO + 3 * np.random.rand() * random.choice((-1, 1))
    # 模拟跳动：偏移坐标系
    geo.offOrigin = np.array([
        8 * np.random.rand() * random.choice((-1, 1)),
        8 * np.random.rand() * random.choice((-1, 1)),
        0
    ])

    angle_two_pri = 0 + 8 * np.random.rand() * random.choice((-1, 1))
    angle_two_sec = 30 + 12 * np.random.rand() * random.choice((-1, 1))
    angles_2_error = np.array([[
        angle_two_pri + 10 * np.random.rand() * random.choice((-1, 1)),
        angle_two_sec + 10 * np.random.rand() * random.choice((-1, 1)),
        0
    ]]) / 180 * np.pi

    projections_2 = tigre.Ax(phantom.copy(), geo, angles_2_error) > 0

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.imshow(projections_2[0], cmap=plt.get_cmap('Greys'))
    plt.savefig(f'./CCTA_second_proj/{save_name}.png')
    plt.close(fig2)  # ⚠️ 释放内存

    # ==========================================
    # 强行叠加生成残次品 3D 考卷
    # ==========================================
    geo.offOrigin = np.array([0, 0, 0])  # 假装不知道动了
    angles_2_theory = np.array([[angle_two_pri, angle_two_sec, 0]]) / 180 * np.pi

    imgSIRT_two = algs.sirt(projections_2, geo, angles_2_theory, 1) > 0

    # 1 + 1 = 2 (生成包含错位和伪影的数据)
    recon = imgSIRT_one.astype(np.int8) + imgSIRT_two.astype(np.int8)
    np.save(f"./CCTA_BP/recon_{save_name}", recon.astype(np.int8))


# =====================================================================
# 脚本运行总开关
# =====================================================================
if __name__ == '__main__':
    # 你可以自由注释掉你不需要运行的模块

    # 步骤 1：处理 ImageCAS 数据集，剥离出 3D Numpy 模型
    #batch_process_imagecas()

    # 步骤 2：对生成的 3D 模型拍 X 光片，并合成带伪影的 3D 输入矩阵
    # (如果上一步还没跑完，记得把下面这行先注释掉)
    batch_generate_projections()