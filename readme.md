# 使用GAN生成手写数字



## 问题概述

通过GAN生成手写数字，训练集来自minst，通过生成网络Generator_Net和判别网络Discriminator_Net进行对抗学习，训练网络。



## 文件介绍

data文件夹中包含了minst数据集。

main程序，其中：

- Generator_Net 输入为N×100，100为随机种子，输出为N×784(28×28); 

- Discriminator_Net 输入为 N×784(28×28)，网络最终输出为N×1，表示判断为真数字的概率；

- 使用 BCEloss 作为误差函数，训练结束后分别保存了生成网络和判别网络的参数。


output文件夹中包含了经过训练后的生成网络生成的数字。



## 参考链接

https://blog.csdn.net/Leytton/article/details/128725434

https://blog.csdn.net/zhuangyuan7838/article/details/121267301