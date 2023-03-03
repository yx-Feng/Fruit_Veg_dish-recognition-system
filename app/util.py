"""
Utilities
"""
import re
import base64
from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """
    将图片从base64格式转换成PIL格式
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    将numpy格式的image (RGB)转换成base64字符串
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

