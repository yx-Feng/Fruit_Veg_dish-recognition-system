import torch.nn as nn 
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms
import numpy as np
import os 
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import time
from model_resnet import ResidualNet
import pandas as pd
import warnings
 
warnings.filterwarnings('ignore')  # 不显示warning

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # 哪些GPU对程序是可见的
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 能报出细节的错误

# ==================================================================
# Constants常量
# ==================================================================
EPOCH = 90                  # epoch的数量，也就是整个数据集训练的次数
BATCH_SIZE = 64              # 每个batch的样本数量
num_workers = 18
LEARNING_RATE = 0.01         # 初始学习率(learning rate)
WEIGHT_DECAY = 0             # 默认的权值衰减(weight decay)
N = 256                      # 输入图片尺寸(512 or 640)
MOMENTUM = (0.9, 0.997)      # Adam优化算法(Adam optimization)中的动量(momentum)
GPU_IN_USE = torch.cuda.is_available()           # 是否使用GPU
DIR_TRAIN_IMAGES = '/hy-tmp/Fruits_Veg_Dishes/train.txt'        # 训练集txt的路径
DIR_TEST_IMAGES = '/hy-tmp/Fruits_Veg_Dishes/val.txt'         # 测试集txt的路径
TRAIN_IMAGE_PATH = '/hy-tmp/Fruits_Veg_Dishes/train'              # 存放images文件的路径
TEST_IMAGE_PATH = '/hy-tmp/Fruits_Veg_Dishes/val'
PATH_MODEL_PARAMS = './model/saved_model.pth'  # 训练好的模型参数的保存路径
NUM_CATEGORIES = 244    # 细粒度类别数量
NUM_COARSE_CATEGORIES = 2  # 粗粒度类别数量
WEIGHT_PATH = '../weights/resnet50-19c8e357.pth'  # 预训练权重
# 两个loss的权重
loss1_weight = 0.7
loss2_weight = 0.3


def My_loader(path):
    return PIL.Image.open(path).convert('RGB')


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, transform=None, target_transform=None, IMAGE_PATH=""):
        data_txt = open(txt_dir, 'r')
        imgs = []

        for line in data_txt:
            line = line.strip()  # 移除字符串头尾的空格或换行符
            words = line.split()  # 分割字符串
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader
        self.IMAGE_PATH = IMAGE_PATH

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]  # 拿到文件名和对应的标签
        try:
            img = self.loader(os.path.join(self.IMAGE_PATH, img_name))  # 拿到图像
            if self.transform is not None:
                img = self.transform(img)  # 转换图像
        except:
            img = np.zeros((256, 256, 3), dtype=float)
            img = PIL.Image.fromarray(np.uint8(img))  # 实现从array到image对象的转换
            if self.transform is not None:
                img = self.transform(img)
            print('error picture:', img_name)
        return img, label


# ==================================================================
# 准备数据集(training & test)
# ==================================================================
print('***** 准备数据 ******')
# 归一化，这里的mean和std用的是Imagenet数据集的均值和标准差，三个值对应三个通道
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize((N, N)),
    transforms.RandomCrop((224, 224)),  # 随机剪裁
    transforms.ToTensor(),
    normalize
])

test_transforms = transforms.Compose([
    transforms.Resize((N, N)),
    transforms.CenterCrop((224, 224)),  # 随机中心剪裁
    transforms.ToTensor(),
    normalize
])

train_dataset = MyDataset(txt_dir=DIR_TRAIN_IMAGES, transform=train_transforms, IMAGE_PATH=TRAIN_IMAGE_PATH)
val_dataset = MyDataset(txt_dir=DIR_TEST_IMAGES, transform=test_transforms, IMAGE_PATH=TEST_IMAGE_PATH)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
print('数据准备 : 完成')

# ==================================================================
# 准备模型
# ==================================================================
print('\n***** 准备模型 *****')

# 构建网络,use_cbam=True表示ResNet50+CBAM,use_cbam=False表示ResNet50
resnet50_cbam = ResidualNet(50, NUM_CATEGORIES, NUM_COARSE_CATEGORIES, use_cbam=True)
pre_weights = torch.load(WEIGHT_PATH)
pre_weights.pop('fc.weight')
pre_weights.pop('fc.bias')
resnet50_cbam.load_state_dict(pre_weights, strict=False)
model = torch.nn.DataParallel(resnet50_cbam)  # 多GPU并行计算

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if GPU_IN_USE:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    model.cuda()
    cudnn.benchmark = True  # 以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题

criterion = nn.CrossEntropyLoss().cuda()  # 使用交叉熵作为损失函数
ignored_params = list()
base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

# 神经网络优化器
optimizer = optim.SGD([
    {'params': base_params}],
    lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
print('模型准备 : 完成')


def train(train_loader, model, criterion, optimizer, epoch):
    global loss1_weight,loss2_weight
    # loss的weight变化
    if epoch == 20:
        loss1_weight = 0.5
        loss2_weight = 0.5
    if epoch == 40:
        loss1_weight = 0.3
        loss2_weight = 0.7
    if epoch == 60:
        loss1_weight = 0
        loss2_weight = 1

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # 训练模式
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # 计算数据加载时间
        data_time.update(time.time() - end)

        # 计算coarse_target,菜品的粗粒度标签为0, 果蔬的粗粒度标签为1
        coarse_target=[]
        for item in target:
            if item < 141:
                coarse_target.append(0)
            else:
                coarse_target.append(1)
        
        coarse_target = torch.tensor(coarse_target).cuda()
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        coarse_target_var = torch.autograd.Variable(coarse_target)
        # 计算输出
        coarse_output, output = model(input_var)
        concate_loss1 = criterion(coarse_output, coarse_target)
        concate_loss2 = criterion(output, target_var)
        loss = loss1_weight*concate_loss1 + loss2_weight*concate_loss2
        # 计算accuracy, 记录loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # 计算gradient, 做一次SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算过去的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 400 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.val, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    global loss1_weight,loss2_weight
    # loss的weight变化
    if epoch == 20:
        loss1_weight = 0.5
        loss2_weight = 0.5
    if epoch == 40:
        loss1_weight = 0.3
        loss2_weight = 0.7
    if epoch == 60:
        loss1_weight = 0
        loss2_weight = 1
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # 评估模式
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # 计算coarse_target,菜品的粗粒度标签为0, 果蔬的粗粒度标签为1
        coarse_target=[]
        for item in target:
            if item < 141:
                coarse_target.append(0)
            else:
                coarse_target.append(1)
                
        coarse_target = torch.tensor(coarse_target).cuda()
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        coarse_target_var = torch.autograd.Variable(coarse_target)
        # 计算输出
        coarse_output, output = model(input_var)
        # 计算loss
        concate_loss1 = criterion(coarse_output, coarse_target)
        concate_loss2 = criterion(output, target_var)
        loss = loss1_weight*concate_loss1 + loss2_weight*concate_loss2
        # 计算accuracy, 记录loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # 计算过去的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 400 == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return losses.val, top1.avg, top5.avg


def save():
    torch.save(model.state_dict(), PATH_MODEL_PARAMS)
    print('Checkpoint saved to {}'.format(PATH_MODEL_PARAMS))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """衰减学习率，每30个epoch衰减10倍"""
    lr = LEARNING_RATE * (0.1 ** (epoch // 30))
    param_groups = optimizer.state_dict()['param_groups']
    param_groups[0]['lr']=lr


def accuracy(output, target, topk=(1,)):
    """计算 precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 开始训练
best_prec1 = 0
# 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
df = pd.DataFrame(columns=['step', 'train_loss', 'train_top1', 'train_top5', 'val_loss', 'val_top1', 'val_top5']) #列名
df.to_csv("./loss_acc/train_val.csv", index=False)  # index表示是否保存索引
for epoch in range(0, EPOCH):
    adjust_learning_rate(optimizer, epoch+1)
    # 训练一个epoch
    train_loss, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch+1)
    # 验证集评价
    val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)
    # 保存acc和loss
    data = pd.DataFrame([[epoch+1, train_loss, train_top1, train_top5, val_loss, val_top1, val_top5]])
    data.to_csv('./loss_acc/train_val.csv', mode='a', header=False, index=False)  # mode='a'表示追加数据了
    # 记录最好的prec@1
    if val_top1 > best_prec1:
        save()
    best_prec1 = max(val_top1, best_prec1)

print(best_prec1)
