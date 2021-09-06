import datetime
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt, cm
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision.models.vgg import VGG
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import h5py
import time
import torch.nn.functional as F

class DataSet(object):
    def __init__(self,filepath):
        file = h5py.File(filepath, 'r')
        self.rgb = file['rgb']
        self.seg = file['seg']
        self.color_codes = file['color_codes']
        self.num_of_classes = len(self.color_codes)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform = transform


    def __getitem__(self, idx):
        img = self.rgb[idx]
        label = self.seg[idx].reshape(128, 256, 1)
        label = label.transpose((2, 0, 1))
        label = label.reshape(128, 256)
        label = torch.FloatTensor(label)
        if self.transform:
          img = self.transform(img)
        return img, label

    def __len__(self):
        return self.rgb.shape[0]

train_dataset =DataSet("train.h5")
test_dataset=DataSet("test.h5")
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
# print(train_dataset.rgb)
# print(train_dataset.num_of_classes)
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(np.array(weight))


vgg16 = models.vgg16(pretrained=True)
pretrained_net = torchvision.models.resnet34(pretrained=True)


vgg16 = models.vgg16(pretrained=True)
num_classes = 34

class FCN(nn.Module):
	def __init__(self):
		super(FCN, self).__init__()
		self.features = vgg16.features
		self.stage1 = nn.Sequential(*list(vgg16.features[:17]))
		self.stage2 = nn.Sequential(*list(vgg16.features[17:24]))
		self.stage3 = nn.Sequential(*list(vgg16.features[24:]))

		self.conv1 = nn.Conv2d(256, num_classes, 1)
		self.conv2 = nn.Conv2d(512, num_classes, 1)
		self.conv3 = nn.Conv2d(512, num_classes, 1)
		self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16,8,4, bias=False)
		self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

		self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
		self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

		self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
		self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel


	def forward(self, x):
		x = self.stage1(x)
		s1 = x
		x = self.stage2(x)
		s2 = x
		x = self.stage3(x)
		s3 = x

		s3 = self.conv3(s3)
		s3 = self.upsample_2x(s3)
		s2 = self.conv2(s2)
		s2 = s2 + s3

		s1 = self.conv1(s1)
		s2 = self.upsample_4x(s2)
		s = s1 + s2

		s = self.upsample_8x(s)
		return s

# class FCN(nn.Module):
# 	def __init__(self):
# 		super(FCN, self).__init__()
# 		self.backbone = models.resnet101(pretrained=True)
# 		self.stage1 = nn.Sequential(*list(vgg16.features[:17]))
# 		self.stage2 = nn.Sequential(*list(vgg16.features[17:24]))
# 		self.stage3 = nn.Sequential(*list(vgg16.features[24:]))
#
# 		self.conv1 = nn.Conv2d(256, num_classes, 1)
# 		self.conv2 = nn.Conv2d(512, num_classes, 1)
# 		self.conv3 = nn.Conv2d(512, num_classes, 1)
# 		self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16,8,4, bias=False)
# 		self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel
#
# 		self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
# 		self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel
#
# 		self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
# 		self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel
#
#
# 	def forward(self, x):
# 		x = self.stage1(x)
# 		s1 = x
# 		x = self.stage2(x)
# 		s2 = x
# 		x = self.stage3(x)
# 		s3 = x
#
# 		s3 = self.conv3(s3)
# 		s3 = self.upsample_2x(s3)
# 		s2 = self.conv2(s2)
# 		s2 = s2 + s3
#
# 		s1 = self.conv1(s1)
# 		s2 = self.upsample_4x(s2)
# 		s = s1 + s2
#
# 		s = self.upsample_8x(s)
# 		return s
BATCH = 10
LR = 5e-6
EPOCHES = 2
WEIGHT_DECAY = 1e-4



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = FCN().to(device)
# print(net)
# 加载数据
# val_data = torch.utils.data.DataLoader(data_val, batch_size=BATCH, shuffle=False)
 # 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)
# 训练时的数据
train_loss = []
train_acc = []
train_acc_cls = []
train_mean_iu = []
train_fwavacc = []

# 验证时的数据
eval_loss = []
eval_acc = []
eval_acc_cls = []
eval_mean_iu = []
eval_fwavacc = []

for e in range(EPOCHES):

    _train_loss = 0
    _train_acc = 0
    _train_acc_cls = 0
    _train_mean_iu = 0
    _train_fwavacc = 0
    loss_all=0
    prev_time = datetime.datetime.now()
    net = net.train()
    for data in train_dataloader:
        if torch.cuda.is_available():
            img = Variable(data[0].cuda())
            label =Variable(data[1].cuda())
        else:
            img = Variable(data[0])
            label =Variable(data[1])
        # 前向传播
        # print(img)
        # print(label)
        with torch.no_grad():
          output = net(img)
          output = F.log_softmax(output, dim=1)
        loss = Variable(criterion(output, label.long()),requires_grad=True)
        loss_all = loss_all + loss.item()
        # 反向传播更新网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _train_loss += loss.item()

        # label_pred输出的是21*224*224的向量，对于每一个点都有21个分类的概率
        # 我们取概率值最大的那个下标作为模型预测的标签，然后计算各种评价指标
        label_pred = output.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()


    net = net.eval()

    _eval_loss = 0
    _eval_acc = 0
    _eval_acc_cls = 0
    _eval_mean_iu = 0
    _eval_fwavacc = 0

    for data in test_dataloader:
        if torch.cuda.is_available():
            img = Variable(data[0].cuda())
            label = Variable(data[1].cuda())
        else:
            img = Variable(data[0])
            label = Variable(data[1])

        # forward
        output = net(img)
        output = F.log_softmax(output, dim=1)
        loss = criterion(output, label.long())
        _eval_loss += loss.item()

        label_pred = output.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()

    # 打印当前轮训练的结果
    cur_time = datetime.datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print( time_str)

