import torch
import numpy as np
from core.datasets.compose import Compose
from utils.checkpoint import load_checkpoint


def init_model(model, data_cfg, device='cuda:0',mode='eval'):
    """从配置文件中初始化一个分类器(classifier)
    Returns:
        nn.Module: The constructed classifier.
    """
    
    if mode == 'train':
        if data_cfg.get('train').get('pretrained_flag') and data_cfg.get('train').get('pretrained_weights'):
            print('Loading {}'.format(data_cfg.get('train').get('pretrained_weights').split('/')[-1]))
            load_checkpoint(model,data_cfg.get('train').get('pretrained_weights'),device,False)

    elif mode == 'eval':
        print('Loading {}'.format(data_cfg.get('test').get('ckpt').split('/')[-1]))
        model.eval()
        load_checkpoint(model,data_cfg.get('test').get('ckpt'),device,False)
        
    model.to(device)
    return model


def inference_model(model, image, val_pipeline, classes_names):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        image (str/ndarray): The image filename or loaded image.
        val_pipeline (dict): The image preprocess pipeline.
        classes_names(list): The classes of datasets.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    if isinstance(image, str):
        if val_pipeline[0]['type'] != 'LoadImageFromFile':
            val_pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=image), img_prefix=None)
    else:
        if val_pipeline[0]['type'] == 'LoadImageFromFile':
            val_pipeline.pop(0)
        data = dict(img=image, filename=None)

    pipeline = Compose(val_pipeline)
    image = pipeline(data)['img'].unsqueeze(0)
    device = next(model.parameters()).device  # model device
    
    # forward the model
    with torch.no_grad():
        scores = model(image.to(device), return_loss=False)
        # top-5预测结果
        order = np.argsort(scores[0])  # 返回的是升序排序后的索引值的数组
        top5 = order[-5:]
        result_json = {}
        for i in range(5):
            category = classes_names[top5[-i - 1]]
            proba = '%.3f' % (scores[0][top5[-i - 1]] * 100)   # 概率保留三位小数
            result_json[category] = str(proba)

    return result_json