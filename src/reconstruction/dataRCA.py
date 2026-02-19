import os

import numpy as np
import torch

#ab_path = os.getcwd() + '/datasets/'

def load_ids(filename):
    # 逻辑：当前脚本(reconstruction) -> 上一级(src) -> datasets
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(os.path.dirname(current_dir), 'datasets')
    path = os.path.join(dataset_root, filename)

    if not os.path.exists(path):
        print(f"⚠️ 找不到 {filename}")
        return []

    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_root = os.path.join(os.path.dirname(current_dir), 'datasets')
        # 定义子文件夹
        self.bp_dir = os.path.join(self.dataset_root, 'CCTA_BP')
        self.gt_dir = os.path.join(self.dataset_root, 'CCTA_GT')
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        data_file = os.path.join(self.bp_dir, 'recon_' + str(ID) + '.npy')
        label_file = os.path.join(self.gt_dir, str(ID) + '.npy')
        # Load data and get label
        #data_file = ab_path + 'CCTA_BP/recon_' + str(ID) + '.npy'
        data = np.transpose(np.load(data_file)[:,:,:,np.newaxis])
        #label_file = ab_path + 'CCTA_GT/' + str(ID) + '.npy'
        label = np.transpose(np.load(label_file)[:,:,:,np.newaxis])

        return torch.from_numpy(data), torch.from_numpy(label)