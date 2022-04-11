# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Generator, Discrimnator
from train import Trainer

batch_size = 128
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
dataset = datasets.MNIST(root='./dataset/', train=True, download=False, transform=transforms)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

generator = Generator()
discrimnator = Discrimnator()

print(generator)
print(discrimnator)

initial_lr = 5e-4
betas = (0.9, 0.99)
g_optimizer = optim.Adam(generator.parameters(), lr=initial_lr, betas=betas)
d_optimizer = optim.Adam(discrimnator.parameters(), lr=initial_lr, betas=betas)

epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(generator, g_optimizer, discrimnator, d_optimizer, device=device)
trainer.train(dataloader, epochs)

name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')