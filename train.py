# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, generator, gen_optim,
                 discriminator, dis_optim,
                 critic_iterations=5, gp_lambda=10,
                 print_every=100, device='cpu'):
        self.device = device

        self.G = generator.to(device)
        self.G_opt = gen_optim
        self.D = discriminator.to(device)
        self.D_opt = dis_optim
        self.critic_iterations = critic_iterations
        self.gp_lambda = gp_lambda
        self.print_every = print_every

        # 对同一噪声在不同训练轮次下的生成结果进行对比
        self.fixed_noise = 2 * (torch.rand(64, 128, device=device) - 0.5)

    def gradient_penalty(self, real_imgs, fake_imgs):
        batch_size = real_imgs.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device).expand_as(real_imgs)
        interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
        interpolated.requires_grad_()

        prediction = self.D(interpolated)

        gradients = autograd.grad(outputs=prediction, inputs=interpolated, grad_outputs=torch.ones_like(prediction),
                                  create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)

        # Avoid vanish
        epsilon = 1e-12
        L2norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + epsilon)
        gp = self.gp_lambda * ((L2norm - 1) ** 2).mean()

        return gp

    # 对于真图, 判别器会输出一个尽可能大的值(由于我们没用到sigmoid函数, 所以输出的不是0~1的概率值)
    # 因此, 生成器的优化策略是送入判别器的假图输出值也尽可能大, 而.backward()的目标是让loss减小, 因此这里我们取负数
    def g_loss_function(self, fake_imgs):
        g_loss = -self.D(fake_imgs).mean()
        return g_loss

    # 判别器的优化策略是让真图和假图的打分差异尽可能的大, 此外需要令梯度惩罚项尽可能小
    def d_loss_function(self, real_imgs, fake_imgs):
        gp = self.gradient_penalty(real_imgs=real_imgs, fake_imgs=fake_imgs)
        d_loss = -(self.D(real_imgs).mean() - self.D(fake_imgs).mean()) + gp
        return d_loss

    def train_single_epoch(self, dataloader):

        d_running_loss = 0
        g_running_loss = 0

        length = len(dataloader)

        for batch_idx, data in enumerate(dataloader, 0):

            real_imgs = data[0].to(self.device)

            for _ in range(self.critic_iterations):
                noise = (torch.rand(real_imgs.size(0), 128, device=self.device) - 0.5) * 2
                fake_imgs = self.G(noise)
                d_loss = self.d_loss_function(real_imgs, fake_imgs)

                self.D_opt.zero_grad()
                d_loss.backward()
                self.D_opt.step()

                d_running_loss += d_loss.item()

            noise = (torch.rand(real_imgs.size(0), 128, device=self.device) - 0.5) * 2
            fake_imgs = self.G(noise)
            g_loss = self.g_loss_function(fake_imgs)

            self.G_opt.zero_grad()
            g_loss.backward()
            self.G_opt.step()

            g_running_loss += g_loss.item()

            if (batch_idx + 1) % self.print_every == 0:
                print('batch:{}/{}, loss(avg.): generator:{}, discriminator:{}'
                      .format(batch_idx + 1,
                              length,
                              d_running_loss/(self.print_every * self.critic_iterations),
                              g_running_loss/self.print_every))

                d_running_loss = 0
                g_running_loss = 0

    @staticmethod
    def save_imgs(images, epoch):
        plt.rcParams['image.cmap'] = 'gray'
        plt.rcParams['figure.figsize'] = (10.0, 8.0)
        plt.rcParams['image.interpolation'] = 'nearest'

        sqrtn = int(np.ceil(np.sqrt(images.size(0))))

        for index, image in enumerate(images):
            plt.subplot(sqrtn, sqrtn, index + 1)
            plt.imshow(image)
            plt.axis('off')
        plt.savefig('./img/Epoch{}.png'.format(epoch+1))

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            print('Epoch:{}'.format(epoch + 1))
            self.train_single_epoch(dataloader)

            with torch.no_grad():
                if epoch + 1:
                    images = self.G(self.fixed_noise).reshape(-1, 28, 28)
                    self.save_imgs(images.cpu(), epoch)
