# -*- coding: utf-8 -*-

import torch.nn as nn


class Discrimnator(nn.Module):
    def __init__(self):
        super(Discrimnator, self).__init__()

        # Input size: Batch Size * Channels * Height * Weight
        #             (for MNIST, the image size is 28 * 28 anf it has only one channel)
        # 输入尺寸: 单批次样本数量 * 通道数 * 图片高度 * 图片宽度 (MNIST的图片尺寸: 单通道 28*28大小)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),     # 输出尺寸: 单批次数量 * 6 * 24 * 24
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),                  # 输出尺寸: 单批次数量 * 6 * 12 * 12
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),    # 输出尺寸: 单批次数量 * 16 * 8 * 8
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),                  # 输出尺寸: 单批次数量 * 16 * 4 * 4
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(3, 3)),  # 输出尺寸: 单批次数量 * 120 * 2 * 2
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),                  # 输出尺寸: 单批次数量 * 120 * 1 * 1
        )

        # Please refer to https://arxiv.org/abs/1701.04862,
        # and there is no need to use a sigmoid function at the end of the discriminator.
        # 参考WGAN论文, 不要在输出层采用sigmoid

        self.full_connection = nn.Sequential(
            nn.Linear(in_features=120, out_features=10),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        x = x.view(-1, 120)  # 由N个120 * 1 * 1的向量变为N个长度为120的向量, 才能输入全连接层
        outputs = self.full_connection(x)
        return outputs


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.full_connection = nn.Sequential(
            nn.Linear(in_features=128, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=2048)
        )

        self.feature_to_img = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, stride=(1, 1), kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_features=4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, stride=(2, 2), kernel_size=(4, 4), padding=(3, 3)),
            nn.BatchNorm2d(num_features=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Tanh()
        )

    def forward(self, noises):
        x = self.full_connection(noises)
        x = x.view(-1, 8, 16, 16)
        fake_imgs = self.feature_to_img(x)
        return fake_imgs
