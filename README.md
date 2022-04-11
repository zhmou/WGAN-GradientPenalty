# WGAN-GradientPenalty
(Chinese Version readme, for English users, see [here](#english-ver))  

一个生成MNIST风格（28 * 28像素, 单通道灰度图）的手写数字图片的demo, 用到了WGAN-GP模型（论文:[Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)）

## 环境
- pytorch 1.10.2
- torchvision 0.11.3
- numpy 1.22.2
- matplotlib 3.5.1

（版本并非强制指定, 这是我自己的版本, 你也可以选用其它版本的库, 但我不能保证一定能正常训练）

## 效果
训练第1个周期:  

<img src="https://user-images.githubusercontent.com/43105172/162693168-15e7aa61-ddaa-41b9-801a-394f8ce9e3bc.png" alt="Epoch1" width="600px">
训练第10个周期:

<img src="https://user-images.githubusercontent.com/43105172/162693222-a3cd0fbc-aaba-4825-9145-73210882ecf6.png" alt="Epoch10" width="600px">
训练第100个周期:

<img src="https://user-images.githubusercontent.com/43105172/162693265-ea435f13-b6c4-42a4-b432-158357014490.png" alt="Epoch100" width="600px">
训练第200个周期:

<img src="https://user-images.githubusercontent.com/43105172/162693334-fcc98c94-8128-4363-8004-70873e20000b.png" alt="Epoch200" width="600px">

## 安装使用
```
git clone https://github.com/zhmou/WGAN-GradientPenalty.git
cd WGAN-GradientPenalty
```
```
python main.py
```
训练时，每一轮次(Epoch)结束后都会在<code>./WGAN-GradientPenalty/img</code>文件夹下生成同一组噪声经生成器(Generator)运算后的结果，在训练完所有轮次后，判别器和生成器的参数将会被保存下来。

## 原理
详参论文，主要是通过梯度惩罚（在判别器的loss中添加了梯度惩罚项，取代了梯度裁剪(weight clipping)）来限制判别器的梯度范围  
至于WGAN的改进，参考[网上的说法](https://zhuanlan.zhihu.com/p/25071913)：
- 判别器最后一层去掉sigmoid函数
- 生成器和判别器的loss不取log
- <del>每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c</del>
- 咱们用的是梯度惩罚
- <del>不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行</del>  
- wgan-gp用的是Adam(原论文)

## 感谢
- https://github.com/EmilienDupont/wgan-gp  
（计算梯度惩罚项的函数是照着写的， 包括几乎整个train.py都是仿造着他写的。但是我实在复现不了他的结果，和这个[Issue](https://github.com/EmilienDupont/wgan-gp/issues/4)里提到的差不多。加之看了下model.py里他还在输出层加了Sigmoid层，似乎有些问题，于是我就自己重写喽）

## English ver.
(to be advised)
A demo for handwritten digits generation (MNIST style: Iamge size: 28 * 28, grayscale) by using WGAN-GP model on pytorch ([Improved Training of Wasserstein GANs](
https://arxiv.org/pdf/1704.00028.pdf))
 
 
