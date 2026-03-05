import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes
import os

# ==========================================
# 🌟 修改这里：填入你下载的本地文件名
# ==========================================
gt_file    = 'gt_392.npy'   # 把 xxx 换成真实 ID
pred_file  = 'pred_392.npy' # 把 xxx 换成真实 ID

def get_3d_mesh(volume, threshold=0.5):
    """提取 3D 表面网格"""
    if np.max(volume) <= threshold:
        print("⚠️ 警告：全是黑的，没有提取到血管！")
        return None, None
    verts, faces, normals, values = marching_cubes(volume, level=threshold)
    return verts, faces

def main():
    if not os.path.exists(pred_file) or not os.path.exists(gt_file):
        print("❌ 找不到 npy 文件，请确保它们和这个 py 文件在同一个文件夹下！")
        return

    print("🔄 正在加载本地 3D 矩阵数据...")
    gt = np.load(gt_file)
    pred = np.load(pred_file)

    print("⚙️ 正在计算 3D 表面网格 (可能需要几秒钟)...")
    verts_gt, faces_gt = get_3d_mesh(gt, threshold=0.5)
    verts_pred, faces_pred = get_3d_mesh(pred, threshold=0.5)

    fig = go.Figure()

    # 🟢 真实血管：半透明绿色，作为背景参考
    if verts_gt is not None:
        fig.add_trace(go.Mesh3d(
            x=verts_gt[:, 0], y=verts_gt[:, 1], z=verts_gt[:, 2],
            i=faces_gt[:, 0], j=faces_gt[:, 1], k=faces_gt[:, 2],
            color='lightgreen', opacity=0.3, name='Ground Truth (Real)'
        ))

    # 🔴 AI 预测血管：不透明红色，高亮显示
    if verts_pred is not None:
        fig.add_trace(go.Mesh3d(
            x=verts_pred[:, 0], y=verts_pred[:, 1], z=verts_pred[:, 2],
            i=faces_pred[:, 0], j=faces_pred[:, 1], k=faces_pred[:, 2],
            color='red', opacity=0.8, name='Prediction (AI)'
        ))

    fig.update_layout(
        title='3D 冠状动脉重建对比 (108 Epochs)',
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    print("🚀 正在直接唤醒浏览器展示 3D 模型...")
    fig.show()  # 魔法就在这一句，直接在本地浏览器中弹开！

if __name__ == '__main__':
    main()