import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
# np.random.seed(515)
plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 23

class tactileDataset(Dataset):
    def __init__(self, data_path, train=True):
        if train:
            self.files = os.listdir(data_path + '/train')
            self.file_path = data_path + '/train/'
        else:
            self.files = os.listdir(data_path + '/test')
            self.file_path = data_path + '/test/'

    def __getitem__(self, index):
        fileName = self.files[index]
        nameStr = fileName.split('_label_')
        label = int(nameStr[-1].split('.')[0])
        data = torch.from_numpy(
            np.load(self.file_path + fileName))  # torch.FloatTensor(np.load(self.file_path + fileName))
        label = torch.LongTensor([label])
        return data, label

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    train_dataset = tactileDataset(r'./data/Ev-Objects', train=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)




