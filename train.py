import torch
from torch.utils.data import DataLoader
import torchvision

import mydataset # 数据集及数据处理
import model    # 网络框架

class Trainer:
    def __init__(self): # 初始化
        # 定义训练运行在什么设备上：cpu或者gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 创建数据集Dataset，然后把Dataset扔进torch的dataloader里
        self.dataset = mydataset.MyDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True,)
        # 构造网络
        self.net = model.MaskRCNN()

        # 训练超参数的设置
        self.optimizer = torch.optim.AdamW() # 优化器，图省事用的Adam系，因为迷信所以用的AdamW
        self.criterion = None

    def train(self): # 训练阶段
        # 设置模型为训练模式
        self.net.train()
        for i, batch in self.dataloader: #
            imgtensor, clstensor, bboxtensor, masktensor = batch
            self.optimizer.zero_grad() # 清空优化器




if __name__ == "__main__":
    # 为了各阶段清晰，所以写成class
    trainer = Trainer()
    trainer.train()  # 训练函数