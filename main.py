import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader


#%% parameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_in_dim = 100  # 生成网络的种子Size=1
G_out_dim = 784 # 生成网络的输出Size=H*W(784=28*28)

D_in_dim = 784  # 判别网络的输入图片Size=H*W(784=28*28)
D_out_dim = 1   # 判别网络输出为真的概率size=1

hidden1_dim = 256
hidden2_dim = 256


#%% net
# 生成网络，负责生成假数据
class Generator_Net(nn.Module):  
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(G_in_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden2_dim, G_out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


# 判别网络，用来判别数据真假
class Discriminator_Net(nn.Module):  
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(D_in_dim, hidden1_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden2_dim, D_out_dim),
            nn.Sigmoid())

    def forward(self, x):
        x = self.layer(x)
        return x


#%% train
epochs = 100
batch_num = 60
lr_rate = 0.0003

data_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5], [0.5])])

train_set = datasets.MNIST(root='data', train=True, transform=data_tf, download=True)
train_loader = DataLoader(train_set, batch_size=batch_num, shuffle=True)

g_net = Generator_Net().to(device)
d_net = Discriminator_Net().to(device)

G_losses = []
D_losses = []

criterion = nn.BCELoss()
G_optimizer = optim.Adam(g_net.parameters(), lr=lr_rate)
D_optimizer = optim.Adam(d_net.parameters(), lr=lr_rate)


iter_count = 0
for epoch in range(epochs):
    print(f'epoch={epoch}/{epochs}')
    for data in train_loader:
        # 生成batch_num张真数字图和假数字图
        img, l = data
        img = img.view(img.size(0), -1).to(device)  # batch_num张真数字图
        r_label = torch.ones(batch_num).to(device)  # batch_num个真标签1
        f_output = g_net(torch.randn(batch_num, G_in_dim).to(device))# batch_num张假数字图
        f_label = torch.zeros(batch_num).to(device) # batch_num个假标签0
        
        
        # 对于真的 判别结果 0~1之间 并计算loss
        r_output = d_net(img)                
        r_loss = criterion(r_output.squeeze(-1), r_label) 
        break
        # 对于假的 判别结果 0~1之间 并计算loss
        d_f_output = d_net(f_output)          
        f_loss = criterion(d_f_output.squeeze(-1), f_label) 
        
        # 优化判别网络，严格区分出真图和生成图
        sum_loss = r_loss + f_loss            
        D_optimizer.zero_grad()
        sum_loss.backward()
        D_optimizer.step()
        
        # 优化生成网络，使得使用判别网络判断生成图约接近真越好
        g_output = g_net(torch.randn(batch_num, G_in_dim).to(device))
        d_output = d_net(g_output)
        d_loss = criterion(d_output.squeeze(-1), r_label)
        G_optimizer.zero_grad()
        d_loss.backward()
        G_optimizer.step()

        if (iter_count % 250 == 0):
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, sum_loss.item(), d_loss.item()))
        iter_count += 1

        G_losses.append(sum_loss.item())
        D_losses.append(d_loss.item())


torch.save(g_net.state_dict(), 'Generator_model.param')
torch.save(d_net.state_dict(), 'Discriminator_model.param')

#%% plot
x = [i for i in range(len(G_losses))]
figure = plt.figure(figsize=(20, 8), dpi=80)
plt.plot(x,G_losses,label='G_losses')
plt.plot(x,D_losses,label='D_losses')
plt.xlabel("iterations",fontsize=15)
plt.ylabel("loss",fontsize=15)
plt.legend()
plt.show()

#%% test

g_net = Generator_Net()
g_net.load_state_dict(torch.load('Generator_model.param'))

with torch.no_grad():
    for i in range(10):
        output = g_net(torch.randn(1, G_in_dim))
        output = output.detach().squeeze().numpy().reshape(28,28)
        plt.figure()
        plt.imshow(output)
        plt.axis("off")
        plt.savefig(f'output/pic-{i}.png')
