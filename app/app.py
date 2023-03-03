from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer
from util import base64_to_pil  # 一些工具
import numpy as np
import torch
from utils.inference import inference_model, init_model
from utils.train_utils import get_info, file2dict
from models.build import BuildNet

# 种类和标签的映射
classes_map = './annotation_chinese.txt'
# 配置文件路径
config = 'models/mobilenet/mobilenet_v3_small.py'
classes_names, _ = get_info(classes_map)
# 从一个配置文件和一个checkpoint文件创建模型
model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config)
# 有GPU就用GPU，没有就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建模型并初始化
model = BuildNet(model_cfg)
model = init_model(model, data_cfg, device=device, mode='eval')

# 声明一个flask应用
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # 主页
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        # 从post request中拿到图像, 读入的图像可能包含RGBA四个通道，这里转成三个通道
        img = base64_to_pil(request.json).convert("RGB")
        # 模型预测
        result_json = inference_model(model, np.array(img), val_pipeline, classes_names)
        return result_json

    return None

if __name__ == '__main__':
    # app.run()  # 云服务器上使用
    # 运行一个web服务器，端口为5000，作为本地测试使用
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print('Check http://127.0.0.1:5000/')
    http_server.serve_forever()
