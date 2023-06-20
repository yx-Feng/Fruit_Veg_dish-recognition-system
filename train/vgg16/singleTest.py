import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models # 使用torchvision自带的模型vgg16
import pandas as pd

# 归一化，这里的mean和std用的是Imagenet数据集的均值和标准差，三个值对应三个通道
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
N = 256
test_transforms = transforms.Compose([
    transforms.Resize((N, N)),
    transforms.CenterCrop((224, 224)),  # 随机中心剪裁
    transforms.ToTensor(),
    normalize
])


def inference_model(model, image, classes_names):

    img = test_transforms(image)
    img2 = torch.unsqueeze(img, 0)  # 给最高位添加一个维度，也就是batchsize的大小
    device = next(model.parameters()).device  # model.parameters()保存的是Weights和Bais参数的值

    # torch.no_grad()表明当前计算不需要反向传播
    with torch.no_grad():
        scores = model(img2.to(device))
        scores = F.softmax(scores, dim=1)
        # top-5预测结果
        order = np.argsort(scores[0].cpu())  # 返回的是升序排序后的索引值的数组
        top5 = order[-5:]
        result_json = {}
        for i in range(5):
            category = classes_names[top5[-i - 1]]
            proba = '%.3f' % (scores[0][top5[-i - 1]] * 100)   # 概率保留三位小数
            result_json[i] = {}
            result_json[i]['id'] = int(top5[-i - 1])
            result_json[i]['name'] = category
            result_json[i]['proba'] = str(proba)

    return result_json, top5

# 拿到类别的中文名称
def get_names(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name_cn, name_en, index = data.split('->')  # name_cn为中文类别名，name_en为英文类别名
        names.append(name_cn)
        indexs.append(int(index))
        
    return names, indexs
  

# 模型权重路径
WEIGHT_PATH = './model/saved_model.pth'
# 种类和标签的映射
classes_map = './annotations-chinese.txt'
# 拿到种类的中文名称
classes_names, _ = get_names(classes_map)
NUM_CATEGORIES = 244    # 类别数量
img = Image.open('./testImages/1.jpg').convert("RGB") # 测试图片路径

# 构建vgg16网络, 使用训练好的权重
model = models.vgg16()
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(4096,NUM_CATEGORIES)
pre_weights = torch.load(WEIGHT_PATH)
# 多GPU并行计算(训练的时候用了，评估的时候也要用，而且要放在load_state_dict前面)
model = torch.nn.DataParallel(model)
model.load_state_dict(pre_weights)
model.eval()
result_json, top5 = inference_model(model, img, classes_names)

df = pd.DataFrame([result_json])
df.to_csv('./result1.csv',sep=',',index=False,header=False)
