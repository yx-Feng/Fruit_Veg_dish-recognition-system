import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

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
        order = np.argsort(scores[0])  # 返回的是升序排序后的索引值的数组
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