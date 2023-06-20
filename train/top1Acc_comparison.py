# 模型top-1 accuracy对比图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# epoch控制为60
EPOCH = 60

x = np.arange(1, EPOCH+1)
# 读取csv中指定列的数据
vgg16_data = pd.read_csv('./vgg16/loss_acc/train_val.csv')
vgg16_val_acc = vgg16_data[['val_top1']]
resnet50_data = pd.read_csv('./resnet50/loss_acc/train_val.csv')
resnet50_val_acc = resnet50_data[['val_top1']]
seresnet50_data = pd.read_csv('./seresnet50/loss_acc/train_val.csv')
seresnet50_val_acc = seresnet50_data[['val_top1']]
resnet50_cbam_data = pd.read_csv('./resnet50+cbam/loss_acc/train_val.csv')
resnet50_cbam_val_acc = resnet50_cbam_data[['val_top1']]

# 将DataFrame类型转化为numpy数组
y1 = np.array(vgg16_val_acc)
y2 = np.array(resnet50_val_acc) 
y3 = np.array(seresnet50_val_acc)
y4 = np.array(resnet50_cbam_val_acc)

# 绘制模型的top-1 accuracy对比图
plt.plot(x, y1[:EPOCH], 'yellow', linewidth=2, label='VGG16')
plt.plot(x, y2[:EPOCH], 'green', linewidth=2, label='ResNet50')
plt.plot(x, y3[:EPOCH], 'blue', linewidth=2, label='SE-ResNet50')
plt.plot(x, y4[:EPOCH], 'red', linewidth=2, label='ResNet50+CBAM')

# x轴和y轴的刻度
plt.xlim(0, 62)
plt.ylim(40, 100)
plt.xlabel('Epochs')
plt.ylabel('Top-1 Accuracy')
plt.legend()
plt.savefig('./top1Acc_comparison.png')
plt.close()  # 关闭窗口