## 果蔬和菜品识别小程序

#### 数据集

融合了以下三个开源数据集。

**Fruits-262**[**链接**](https://www.kaggle.com/datasets/aelchimminut/fruits262)。

**Vegetable Image Dataset**[**链接**](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset?resource=download)：。

**ChineseFoodNet：**[**官网**](https://sites.google.com/view/chinesefoodnet) 。

#### 代码

**train目录**：存放训练模型的代码。分别使用resnet50、resnet50+cbam、seresnet50、vgg16四个模型进行训练

**frontend目录**：小程序前端代码。weui+原生API

**backend目录**：小程序后台代码。flask。模型放在backend/model目录。

## 部署见博客

**博客**：[https://fengyongxuan.blog.csdn.net/article/details/128975057](https://fengyongxuan.blog.csdn.net/article/details/128975057)
