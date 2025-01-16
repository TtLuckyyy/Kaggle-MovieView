# 该文件用于处理数据集
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
class MovieDataset(Dataset):
    def __init__(self, train, max_length=128):
        """
        :param train: 是否是训练集，True 是训练集，False 是验证集
        :param max_length: 每个文本的最大长度（用于填充和截断）
        """
        # 构建数据样本
        self.train = train
        self.max_length = max_length

        # 文件路径检查
        file_path = 'data/train.tsv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 '{file_path}' 未找到")

        # 读取数据
        self.data = pd.read_csv(file_path, sep='\t')

        if self.train:
            # 划分 80% 训练集，20% 验证集
            self.data, _ = train_test_split(self.data, test_size=0.2, random_state=1)
        else:
            # 使用剩余的 20% 数据作为验证集
            _, self.data = train_test_split(self.data, test_size=0.2, random_state=1)

        # 重新生成索引
        self.data = self.data.reset_index(drop=True)

        # 训练集或验证集长度
        self.len = self.data.shape[0]

        # 提取特征和标签
        self.x_data, self.y_data = self.data['Phrase'], self.data['Sentiment']

    def __getitem__(self, index):
        # 根据数据索引获取样本
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回数据长度
        return self.len