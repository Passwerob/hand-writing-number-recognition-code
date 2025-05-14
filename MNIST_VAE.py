import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决某些系统上的OpenMP冲突问题

# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):


        super(Encoder, self).__init__()
        # 第一层全连接层，将输入映射到隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # 输出层，分别输出均值(mu)和方差的对数(logvar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x: 输入图像数据
            
        Returns:
            mu: 潜在空间的均值
            logvar: 潜在空间方差的对数
        """
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# 定义解码器网络
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        初始化解码器网络
        
        Args:
            latent_dim: 潜在空间维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（与输入维度相同，784）
        """
        super(Decoder, self).__init__()
        # 第一层全连接层，将潜在空间映射到隐藏层
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()
        # 输出层，将隐藏层映射回原始维度
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 使用sigmoid确保输出在[0,1]范围内

    def forward(self, z):
        """
        前向传播过程
        
        Args:
            z: 潜在空间采样点
            
        Returns:
            x_recon: 重建的图像数据
        """
        h = self.relu(self.fc1(z))
        x_recon = self.sigmoid(self.fc2(h))
        return x_recon


# 定义变分自编码器（VAE）模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        初始化VAE模型
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            latent_dim: 潜在空间维度
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧，用于从编码器输出的分布中采样
        
        Args:
            mu: 均值
            logvar: 方差的对数
            
        Returns:
            z: 采样得到的潜在变量
        """
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)    # 从标准正态分布采样
        return mu + eps * std          # 重参数化

    def forward(self, x):
        """
        VAE的前向传播过程
        
        Args:
            x: 输入图像数据
            
        Returns:
            x_recon: 重建的图像
            mu: 潜在空间均值
            logvar: 潜在空间方差的对数
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def loss_function(x_recon, x, mu, logvar):
    """
    计算VAE的损失函数，包括重建损失和KL散度
    
    Args:
        x_recon: 重建的图像
        x: 原始图像
        mu: 潜在空间均值
        logvar: 潜在空间方差的对数
        
    Returns:
        total_loss: 总损失值
    """
    # 计算重建损失（二进制交叉熵）
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
    # 计算KL散度（正则化项）
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# 设置超参数
input_dim = 784      # MNIST图像展平后的维度（28*28）
hidden_dim = 400     # 隐藏层维度
latent_dim = 20      # 潜在空间维度
batch_size = 128     # 批次大小
epochs = 10          # 训练轮数
learning_rate = 1e-3 # 学习率

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor()  # 将PIL图像转换为张量，并归一化到[0,1]
])

# 加载MNIST训练数据集
train_dataset = datasets.MNIST(root='./data', train=True,
                             transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # 将图像展平
        data = data.view(-1, 784)
        optimizer.zero_grad()
        
        # 前向传播
        x_recon, mu, logvar = model(data)
        # 计算损失
        loss = loss_function(x_recon, data, mu, logvar)
        
        # 反向传播和优化
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    # 打印每个epoch的平均损失
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader.dataset)}')

# 生成新样本
with torch.no_grad():
    # 从标准正态分布采样潜在变量
    z = torch.randn(16, latent_dim)
    # 使用解码器生成图像
    sample = model.decoder(z)
    # 保存生成的样本
    vutils.save_image(sample.view(16, 1, 28, 28), 'samples.png', normalize=True)

# 显示生成的样本
img = plt.imread('samples.png')
plt.imshow(img)
plt.axis('off')
plt.show()
