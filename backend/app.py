from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer
from io import BytesIO
from PIL import Image
import torch
from flask_cors import CORS
import pymysql
from model.model_resnet import ResidualNet
from utils.get_info import get_names
from utils.inference import inference_model
from utils.histogram_equalization import hisEqulColor

# 模型权重路径
WEIGHT_PATH = './model/saved_model.pth'
# 种类和标签的映射
classes_map = './annotations-chinese.txt'
# 拿到种类的中文名称
classes_names, _ = get_names(classes_map)
NUM_CATEGORIES = 244    # 类别数量
# 有GPU就用GPU，没有就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建resnet50+cbam网络, 使用训练好的权重
resnet50_ = ResidualNet(50, NUM_CATEGORIES, use_cbam=True)
pre_weights = torch.load(WEIGHT_PATH, map_location=torch.device(device))
# 多GPU并行计算(训练的时候用了，评估的时候也要用，而且要放在load_state_dict前面)
model = torch.nn.DataParallel(resnet50_)
model.load_state_dict(pre_weights)
model.eval()

# 声明一个flask应用
app = Flask(__name__)
# 允许跨域，supports_credentials=True表示允许请求发送cookie
CORS(app, supports_credentials=True)

# 打开数据库连接
conn = pymysql.connect(
    host='localhost',  # MySQL服务器地址
    user='root',  # 用户名
    password='123456',  # 密码
    charset='utf8',
    port=3306,  # 端口
    db='fruit_veg_dish',  # 数据库名称
)
# 使用cursor()方法获取操作游标
c = conn.cursor()


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        # 传过来的是一个filestorage对象
        img_bytes = BytesIO(request.files['image'].read())
        # 读入的图像可能包含RGBA四个通道，这里转成三个通道，返回PIL类型的数据
        img = Image.open(img_bytes).convert("RGB")
        # 使用CLAHE算法进行光照预处理
        img = hisEqulColor(img)
        # img = base64_to_pil(request.files['image'].read()).convert("RGB")
        # 模型预测
        result_json, top5 = inference_model(model, img, classes_names)
        for i in range(5):
            # c.fetchall[0]拿到对应id种类的描述
            sql = "select description from info where id={}".format(int(result_json[i]['id']))
            # 检查连接是否断开，如果断开就进行重连
            conn.ping(reconnect=True)
            c.execute(sql)
            result_json[i]['description'] = c.fetchall()[0][0]
        # print(result_json)
        return {'result': result_json}

    return None


if __name__ == '__main__':
    # app.run()  # 云服务器上使用
    # 运行一个web服务器，端口为5000，作为本地测试使用
    http_server = WSGIServer(('0.0.0.0', 5005), app)
    print('Check http://127.0.0.1:5005/')
    http_server.serve_forever()
