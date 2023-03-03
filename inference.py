#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from importlib import import_module


class TagPytorchInference(object):

    # **kwargs表示关键字列表，本质是一个dict
    def __init__(self, **kwargs):
        _input_size = kwargs.get('input_size', 224)  # 默认input_size参数为224，也可以自己指定
        self.input_size = (_input_size, _input_size)
        self.num_classes = kwargs.get('num_classes', 309)  # 默认num_classes参数为309
        kwargs['num_classes'] = self.num_classes
        self.gpu_index = kwargs.get('gpu_index', '0')  # 默认gpu_index参数为0
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_index  # 指定所要使用的显卡
        self.net = self._create_model(**kwargs)  # 创建模型
        self._load(**kwargs)
        self.net.eval()  # 不启用BatchNormalization和Dropout
        self.transforms = transforms.ToTensor()  # 转换为pytorch可以快速处理的张量格式
        if torch.cuda.is_available():
            self.net.cuda()

    def close(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空显存缓冲区

    def _create_model(self, **kwargs):
        module_name = kwargs.get('module_name', 'mobilenet_v2_module')  # module默认使用mobilenet_v2_module
        net_name = kwargs.get('net_name', 'mobilenet_v2')  # net名默认为mobilenet_v2
        m = import_module('nets.' + module_name)  # 加载net文件夹下面的module
        model = getattr(m, net_name)  # 返回m对象的属性值
        net = model(**kwargs)
        return net

    def _load(self, **kwargs):
        current_folder = os.path.dirname(__file__)  # 显示当前文件
        _model_name = os.path.join(current_folder, 'model', 'CWFood_model.pth')
        model_name = kwargs.get('model_name', _model_name)
        model_filename = model_name
        state_dict = torch.load(model_filename, torch.device('cpu'))
        self.net.load_state_dict(state_dict)

    def image_preproces(self, image_data):
        _image = cv2.resize(image_data, self.input_size)
        # _image = _image[:,:,::-1]   # bgr2rgb
        return _image.copy()

    def run(self, image_data, **kwargs):
        _image_data = self.image_preproces(image_data)   # 图像预处理
        input = self.transforms(_image_data)
        _size = input.size()
        input = input.resize_(1, _size[0],_size[1],_size[2])  # 改变input tensor的形状
        if torch.cuda.is_available():
            input = input.cuda()  # 使数据在GPU上进行运算
        logit = self.net(Variable(input))
        # softmax
        infer = F.softmax(logit, 1)
        return infer.data.cpu().numpy().tolist()

# 模型预测
def model_predict(image):
    if image is None:
        raise TypeError('image data is none')
    tagInfer = TagPytorchInference()
    result = tagInfer.run(image)
    tagInfer.close()
    # top-5
    order = np.argsort(result[0])  # 返回的是升序排序后的索引值的数组
    top5 = order[-5:]
    return result, top5
