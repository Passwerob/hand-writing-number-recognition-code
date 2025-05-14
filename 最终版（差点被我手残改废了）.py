import torch  # 配置Pytorch深度学习框架
import torch.nn as nn  # 配置神经网络模块的PyTorch子库
import torch.optim as optim  # 配置优化算法的PyTorch子库。
from torch.utils.data import Dataset, DataLoader
# torch.utils.data: 一个模块，包含数据处理相关的类和函数。
# Dataset, DataLoader: 这是从torch.utils.data模块中导入的两个类。
import numpy as np  # 用来存储和处理大型多维矩阵
import struct  # struct用于读取二进制数据
import matplotlib.pyplot as plt  # 借助matplotlib库显示预测结果和图片
from torchvision import transforms  # 从 torchvision 库中导入 transforms 模块


# 从torchvision中导入transforms  ，即torchvision库中的图像转换工具

class MNISTDataset(Dataset):  # class 用于定义类 此处为MNIST数据集
    def __init__(self, image_path, label_path, transform=None):
        # def用于定义函数或方法
        # init初始方法读取图像数据和信息（创建图像类）
        # transform=None表示不进行任何可选的图像转换操作，即使用初始数据
        self.transform = transform  # 将传入的transform参数赋值给当前对象的transform属性。
        with open(image_path, 'rb') as f:  # 将打开的文件对象赋值给变量f ；rb表示以二进制打开，类似的还有r，r+等
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            # struct.unpack用于将二进制数据转换为Python数据类型
            # ">IIII": 格式字符串，指定了数据的类型和顺序，'>'表示大端模式，'I'表示无符号整数。
            # f.read(16): 从文件中读取16个字节。
            self.images = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
            # np.fromfile: NumPy函数，用于从二进制文件中读取数据。
            # dtype=np.uint8: 指定读取数据的类型为无符号8位整数。
            # .reshape(num, 784): 将读取的一维数组重塑为指定的形状，num是图像数量，784是每张图像的像素数（28x28）。

            # 图像文件格式
            # 对于MNIST图像文件（例如train-images.idx3-ubyte）：
            # 前16字节：文件头，包含4个无符号整数（>IIII）：
            # 第0字节：魔术数字（Magic Number），用于标识文件类型，通常是2051。
            # 第4字节：图像数量
            # 第8字节：图像的行数
            # 第12字节：图像的列数
            # 接下来的字节 实际的图像数据
            # 标签文件格式
            # 对于MNIST标签文件（例如train-labels.idx1-ubyte）
            # 前8字节：文件头，包含2个无符号整数（>II）
            # 第0字节：魔术数字，用于标识文件类型，通常是2049
            # 第4字节：标签数量（Number of labels）
            # 接下来的字节：实际的标签数据。
        with open(label_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))  # 以大端模式读取数据，读取文件的前8个字节，并将其转换为2个无符号整数
            self.labels = np.fromfile(f, dtype=np.uint8)  # ，通过文件对象f，，文件对象f允许打开文件，从文件中读取标签数据
            # np.uint8 是一个数据类型，表示无符号的8位整数，范围从0到255

    def __len__(self):
        return len(self.labels)  # 使用len方法获取对象长度 在这里返还标签数组的长度，也就是数据集中样本的数量

    def __getitem__(self, idx):  # getitem用于获取数据集里的单个样本
        image = self.images[idx]  # 获得索引为idx的图像信息
        label = self.labels[idx]  # 获得索引为idx的标签信息
        image = image.reshape(28, 28)  # 将图像数据重塑为28x28的形状
        if self.transform:  # 检查是否定义了转换操作，一开始在定义MNIST数据集时，我已经定义了transform，所以这里时True
            image = torch.tensor(image, dtype=torch.float) / 255.0  # 将图像数据转换为张量，并归一化到[0, 1]区间。
            image = transforms.ToPILImage()(image)  # 将张量转化为PIL图像
            image = self.transform(image)  # 将之前定义的一系列转换操作运用到PIL图片上
        return image, label  # 返回图像和标签


# 数据转换
transform = transforms.Compose([  # 将所调出来多个图片转换拼接为一个
    transforms.ToTensor(),  # 将对象转换为Pytorch张量（Tensor）
    # 就本次而言为PIL图像(灰度图)，单通道，模式通常是“高度 x 宽度 x 通道”，并且值的范围是 [0, 255]
    # 对NumPy数组的形状通常是“通道 x 高度 x 宽度”（对于RGB图像，通道是3），并且值的范围也是 [0, 255]。
    transforms.Normalize((0.5,), (0.5,))  # 对张量进行归一化处理，使数据的均值mean为0.5，标准差std为0.5
])  # 归一化操作，有助于提高模型的训练速度和收敛效率

# 第一次实例化数据集
train_dataset = MNISTDataset(
    image_path=rb'C:\Users\Richard\Desktop\train-images.idx3-ubyte',
    label_path=rb'C:\Users\Richard\Desktop\train-labels.idx1-ubyte',
    transform=transform
)
# 这里的实例化是为了创建一个数据集，其包含了测试图像及其对应的标签。这个数据集对象随后被用来创建一个 DataLoader 对象，该对象可以在测试过程中批量地提供数据给模型。（为了后面的输出做准备）
# 第一次引入全局数据集，利用rb从指定绝对路径，读取二进制数据，将获得的transform数据赋给transform

# 加载训练数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Dataloader即数据迭代器，batch_size=1024
# shuffle=True 在训练时使用，打乱原始数据，提高模型泛化能力
# 同理shuffle=False 在测试与验证的时候被使用，确保按照原始数据处理

class Net(nn.Module):  # 实例化Net类，创建模型对象，继承自nn.Module
    def __init__(self):  # 类的构造函数
        super(Net, self).__init__()  # 调用父类的构造方法，确保父类函数被正确的初始化
        self.fc1 = nn.Linear(28 * 28, 128)  # 建立第一个全连接层（线性层或仿射变换层） 它有 28*28 个输入特征（因为 MNIST 数据集的每个图像是 28x28 的）和 128 个输出特征
        self.fc2 = nn.Linear(128, 64)  # 定义了第二个全连接层，它接收来自 fc1 的 128 个特征，并输出 64 个特征。
        self.fc3 = nn.Linear(64, 10)  # 定义了第三个全连接层，它接收来自 fc2 的 64 个特征，并输出 10 个特征（对应于 MNIST 数据集中的 10 个类别）

    def forward(self, x):  # 前向传播函数  PyTorch 会自动调用这个函数进行前向传播
        x = x.view(-1, 28 * 28)  # 展平输出： 输入 x 重塑为一个二维张量  -1 表示自动计算该维度的大小 ，往后使用ReLU激活函数
        x = torch.relu(self.fc1(x))  # 第一个全连接层：将输入 x 通过第一个全连接层 fc1 ，应用 ReLU激活函数
        x = torch.relu(self.fc2(x))  # 第二个全连接层：将fc1 的输出通过第二个全连接层 fc2，再次应用 ReLU 激活函数
        x = self.fc3(x)  # 第三个全连接层：将 fc2 的输出通过第三个全连接层 fc3。在这个层之后不使用激活函数，因为输出层通常直接连接到损失函数
        # 输出层的激活函数通常是 softmax，它在计算交叉熵损失时隐式地被应用
        return x  # 返还经过三个全连接层处理的图像数据，此时x为多维矩阵


model = Net()  # Net 类的一个实例被创建并赋值给变量 model

criterion = nn.CrossEntropyLoss()  # 运用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)
# 随机梯度下降优化器SGD
# lr为学习率 学习率过小，每次迭代效果不明显：学习率过大，可能会出现震荡过强或者过拟合
# 借助momentum（动量）算法，动量衰减系数γ为0.9，在每次迭代中，当前梯度的90%将与上一时刻的动量值相加，然后以这个结果来更新参数，加速梯度下降
for epoch in range(10):  # 进行10个epoch的训练
    for images, labels in train_loader:  # 由train_loader提供图像标签
        optimizer.zero_grad()  # 归零上一个epoch更新中积累的梯度
        outputs = model(images)  # 执行我所构建模型的前向传播
        loss = criterion(outputs, labels)  # loss为所计算出的预测输出和真实标签之间的损失
        loss.backward()  # 执行反向传播。计算梯度
        optimizer.step()  # 根据梯度，更新我所构建神经网络的参数

    print(f'Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}')  # 将每个epoch的运行进程显示出来并打印出LOSS
    # 第二次实例化测试数据集
    test_dataset = MNISTDataset(
        image_path=rb'C:\Users\Richard\Desktop\train-images.idx3-ubyte',
        label_path=rb'C:\Users\Richard\Desktop\train-labels.idx1-ubyte',
        transform=transform
    )
    # MNISTDataset类的一个实例被创建为一个数据集对象，用于测试数据
    # 同样创建一个数据集对象，但是这次是为了从测试集中随机选择一些图像
    # 并使用模型进行预测。预测的结果将被用来显示图像及其预测的类别。

    # 加载测试数据
    test_loader = DataLoader(test_dataset, batch_size=6480, shuffle=False)  # 用于批量加载测试数据

    model.eval()  # 设置模型为评估模式（这会禁用Dropout层和Batch Normalization等层）
    with torch.no_grad():  # 将梯度计算禁用（因此不会更新模型的权重），节省内存，加速计算
        correct = 0  # 用于记录模型中预测正确的图像数量
        total = 0  # 用于记录测试集中图像的总数
        for images, labels in test_loader:  # 从test_loader按照批次抽取图片和标签
            outputs = model(images)  # 调用模型进行前向传播，计算一批图像的预测输出
            _, predicted = torch.max(outputs.data, 1)  # 在维度1查找每个元素的最大值，使用_来忽略最大值，从而仅仅获取预测的类别Pred
            total += labels.size(0)  # 获取当前批次的数量，并累加到total
            correct += (predicted == labels).sum().item()
            #  predicted == labels 创建一个布尔张量
            # 其中 True 表示预测与真实标签相同（即正确预测）
            #  False 表示预测与标签不同
            # .sum()计算布尔张量中True的数量，并累加到correct（即实现了正确预测的数量的累加）
            # .item()表示从单个元素张量中提取出 Python 数字，用于累加
        print(f'Accuracy of the model on the test images: {100 * correct / total}%')
        # 打印模型在测试数据集上的准确率


def load_images(image_path, num_images=5):
    # 定义一个函数load_images，来加载指定数量的图像
    # 接受两个参数：image_path：图像文件的路径
    #           num_images：要加载的图像数量，在这里我设置为5
    with open(image_path, 'rb') as f:  # 同样的with open语句 同样的rb二进制读取
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # 同样的struct.unpack来将二进制数据转换为Python数据类型
        # 同样的格式字符串，指示unpack解释字节数据
        # 同样f.read(16) 读取文件的前 16 个字节
        # magic 是一个魔术数字，用于标识文件类型。
        # num 是文件中图像的总数。
        # rows 和 cols 分别是图像的高度和宽度
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
        # 同样的np.fromfile 函数读取二进制文件中的数据，并将其转换为NumPy数组
        # 同样的dtype=np.uint8 指定数组元素的类型为无符号8位整数
        # .reshape(num, 784) 将一维数组重塑为二维数组 其中每行对应一个图像 共有 num 行
        # 每个图像有 784 个像素（即28x28像素的手写数字图像被拉平为一维数组）

    # 随机选择num_images张图片
    indices = np.random.choice(num, num_images, replace=False)
    # np.random.choice 函数从 num 个图像中随机选择 num_images 个不重复的索引
    # replace=False 确保不会选择重复的图像
    selected_images = images[indices]
    # 通过索引数组选择图像，并将这些图像存储在 selected_images 中

    return selected_images, indices
    # 函数返回两个值 selected_images：一个包含所选图像的数组
    #                    indices：所选图像在原始数据集中的索引数组


# 加载5张测试图片
test_image_path = rb'C:\Users\Richard\Desktop\t10k-images.idx3-ubyte'  # 定义测试图像文件的路径
images, indices = load_images(test_image_path)
# 调用load_images 函数 确保从指定路径加载图像
# 返回的图像数据和它们的索引被分别存储在 images 和 indices 中

# 将图片转换为PyTorch张量并归一化
images_tensor = torch.tensor(images, dtype=torch.float) / 255.0
# torch.tensor 将 NumPy 数组 images 转换为 PyTorch 张量
# dtype=torch.float 指定张量的数据类型为浮点数
# / 255.0 将图像的像素值从 [0, 255] 归一化到 [0.0, 1.0]
images_tensor = images_tensor.view(-1, 28 * 28)  # 将张量展平，每个图像成为一维数组
# 同样的-1表示自动计算该维度的大小
# 模型预测
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 同样禁用梯度计算
    outputs = model(images_tensor)  # 前向传播，计算输入图像的预测输出
    _, predicted = torch.max(outputs.data, 1)  # 获取预测结果，与上方形式一样

# 打印结果
for i in range(len(predicted)):  # 遍历预测结果
    print(f'Image Index: {indices[i]}, Predicted Label: {predicted[i].item()}')  # 打印每张图片的索引和模型预测的类别。
    # indices[i]获取第 i 张图像的索引
    # predicted[i].item() 获取第 i 个预测标签的值
    # 打印每张图像的索引和模型预测的类别
# 显示图片及其预测结果
plt.figure(figsize=(28, 28))  # 创建一个新的Matplotlib图形，设置图形的大小
for i in range(len(predicted)):  # 遍取预测结果
    plt.subplot(1, len(predicted), i + 1)  # 创建子图
    # 第一个参数args表示子图在图形中的行数
    # 第二个参数len（predicted）表示子图在图形中的列数
    # 第三个参数 i + 1 表示当前子图的索引
    plt.imshow(images[i].reshape(28, 28), cmap='gray')
    # plt.title() 设置当前子图的标题
    # 显示图片，camp=‘gray’表示以灰度模式显示
    plt.title(f'Pred: {predicted[i].item()}')
    # 设置子图的标题，显示预测的类别
    # f'Pred: {predicted[i].item()}' 使用 f-string 格式化字符串
    # 显示 "Pred: " 后跟预测的类别。predicted[i].item()
    # 从 PyTorch 张量中提取第 i 个预测类别的标量值
    plt.axis('off')
    # off表示关闭子图的坐标轴，使图像显示的时候不会有坐标轴
plt.show()  # 显示最终的图形
