from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer
from util import base64_to_pil  # 一些工具
from inference import model_predict
import numpy as np

# 声明一个flask应用
app = Flask(__name__)

CLASSES = [
    'Mapo_Tofu', 'Home_style_sauteed_Tofu', 'Fried_Tofu', 'Bean_curd', 'Stinky_tofu', 'Potato_silk', 'Pan_fried_potato', 'Pan_fried_potato', 'Braised_beans_with_potato', 'Fried_Potato_Green_Pepper_&_Eggplant',
    'French_fries', 'Yu-Shiang_Eggplant', 'Mashed_garlic_eggplant', 'Eggplant_with_mince_pork', 'Spicy_cabbage', 'Sour_cabbage', 'Steamed_Baby_Cabbage', 'Shredded_cabbage', 'Sauteed_Lettuce_in_Oyster_Sauce', 'Saute_vegetable',
    'Tumis_kangkung', 'Lettuce_with_smashed_garlic', 'Sauteed_spainch', 'Sauteed_bean_sprouts', 'Sauteed_broad_beans', 'Soybean', 'Broccoli_with_Oyster_Sauce', 'Deep_Fried_lotus_root', 'Lotus_root', 'Tomato_salad',
    'Gizzard', 'Black_Fungus_in_Vinegar_Sauce', 'Cucumber_in_Sauce', 'Peanut', 'Seaweed_salad', 'Chinese_Yam_in_Hot_Toffee', 'Fried_Yam', 'Fried_beans', 'Oyster_mushroom', 'Stuffed_bitter_melon',
    'Sauteed_bitter_melon', 'Pepper_with_tiger_skin', 'Yuba_salad', 'Fried_cauliflower', 'Sauteed_Sweet_Corn_with_Pine_Nuts', 'Sauted_Chinese_Greens_with_Mushrooms', 'Spiced_mushroom', 'Celery_and_tofu', 'Sauteed_Lily_Bulbs_and_Celery', 'Leak_and_tofu',
    'Scrambled_egg_with_tomato', 'Scrambled_Egg_with_Leek', 'Scrambled_Egg_with_cucumber', 'Steamed_egg_custard', 'Pork_liver', 'Pig_ears', 'Roast_pork', 'Steamed_pork_with_rice_powder', 'Sweet_and_sour_spareribs', 'Braised_spareribs_with_kelp',
    'Cola_Chicken_wings', 'Chicken_Feet_with_Pickled_Peppers', 'Chicken_Feet_with_black_bean_sauce', 'Steamed_Chicken_with_Chili_Sauce', 'Roast_goose', 'Boiled_chicken', 'Saute_Spicy_Chicken', 'Steamed_Chicken_with_Mushroom', 'Chicken_braised_with_brown_sauce', 'Soy_sauce_chicken',
    'Spicy_Chicken', 'Kung_Pao_Chicken', 'Stewed_Chicken_with_Three_Cups_Sauce', 'Shredded_chicken', 'Fried_chicken_drumsticks', 'Beer_duck', 'Scalloped_pork_or_lamb_kidneys', 'Braised_pork', 'Braised_beef', 'Beef_Seasoned_with_Soy_Sauce_',
    'Sirloin_tomatoes', 'Stewed_sirloin_potatoes', 'Sauteed_Beef_Fillet_with_Hot_Green_Pepper', 'Pork_with_salted_vegetable', 'Double_cooked_pork_slices', 'Braised_Pork_with_Vermicelli', 'Boiled_Shredded_pork_in_chili_oil', 'Fried_Sweet_and_Sour_Tenderloin', 'Cripsy_sweet_&_sour_pork_slices', 'Pot_bag_meat',
    'Shredded_Pork_with_Vegetables', 'Tiger_lily_buds_in_Baconic', 'Sauteed_Shredded_Pork_in_Sweet_Bean_Sauce', 'Shredded_pork_with_bean', 'Braised_pig_feet_with_soy_sauce', 'Tripe', 'Shredded_pork_and_green_pepper', 'Yu-Shiang_Shredded_Pork_', 'Braised_Fungus_with_pork_slice', 'Sauteed_Sliced_Pork_Eggs_and_Black_Fungus',
    'Lettuce_shredded_meat', 'Sauteed_Vermicelli_with_Spicy_Minced_Pork', 'Fried_Lamb_with_Cumin', 'Lamb_shashlik', 'Sauteed_Sliced_Lamb_with_Scallion', 'Stewed_Pork_Ball_in_Brown_Sauce', 'Boiled_Fish_with_Picked_Cabbage_and_Chili', 'Grilled_fish', 'Sweet_and_sour_fish', 'Sweet_and_Sour_Mandarin_Fish',
    'Braised_Hairtail_in_Brown_Sauce', 'Steamed_Fish_Head_with_Diced_Hot_Red_Peppers', 'Fish_Filets_in_Hot_Chili_Oil', 'Steamed_Perch', 'Cheese_Shrimp_Meat', 'Shrimp_broccoli', 'Braised_Shrimp_in_chili_oil', 'Spicy_shrimp_', 'Spicy_crayfish', 'Shrimp_Duplings',
    'Steamed_shrimp_with_garlic_and_vermicelli', 'Sauteed_Shrimp_meat', 'Pipi_shrimp', 'Scallop_in_Shell', 'Oysters', 'Squid', 'Abalone', 'Crab', 'Turtle', 'Eel',
    'Yangzhou_fried_rice', 'Omelette', 'Steamed_Bun_Stuffed', 'Steamed_Pork_Dumplings', 'Egg_omelet', 'Potato_omelet', 'Egg_pie_cake', 'Marinated_Egg', 'Poached_Egg', 'Pine_cake_with_Diced_Scallion',
    'Sesame_seed_cake', 'Chinese_hamburger', 'Leek_box', 'Steamed_bun_with_purple_potato_and_pumpkin', 'Steamed_bun', 'Steamed_stuffed_bun', 'Pumpkin_pie', 'Pizza', 'Deep-Fried_Dough_Sticks', 'Sauteed_noodles_with_minced_meat',
    'Chongqing_Hot_and_Sour_Rice_Noodles', 'Cold_noodles', 'Noodles_with_egg_and_tomato', 'Spaghetti_with_meat_sauce', 'Noodles_with_tomato_sauce', 'Cold_Rice_Noodles', 'Sichuan_noodles_with_peppery_sauce', 'Qishan_noodles', 'Fried_noodles', 'Dumplings',
    'Corn_Cob', 'Braised_beef_noodle', 'fried_rice_noodles', 'Steamed_vermicelli_roll', 'Pork_wonton', 'Fried_Dumplings', 'Tang-yuan', 'Millet_congee', 'Sweet_potato_porridge', 'Jellyfish',
    'Minced_Pork_Congee_with_Preserved_Egg', 'Rice_porridge', 'Rice', 'Laver_rice', 'Stone_pot_of_rice', 'Black_bone_chicken_soup', 'Crucian_and_Bean_Curd_Soup', 'Dough_Drop_and_Assorted_Vegetable_Soup', 'Hot_and_Sour_Soup', 'Pork_ribs_soup_with_radish',
    'Tomato_and_Egg_Soup', 'West_Lake_beef_soup', 'Lotus_Root_and_Rib_soup', 'Seaweed_and_Egg_Soup', 'Seaweed_tofu_soup', 'Corn_and_sparerib_soup', 'Spinach_and_pork_liver_soup', 'Borsch', 'White_fungus_soup', 'White_gourd_soup',
    'Miso_soup', 'Duck_Blood_in_Chili_Sauce', 'Pork_Lungs_in_Chili_Sauce', 'Spicy_pot', 'Golden_meat_rolls', 'Chiffon_Cake', 'Egg_Tart', 'Bread', 'Croissant', 'Toast',
    'Biscuits', 'Cookies', 'Soda_biscuit', 'Double_skin_milk', 'Ice_cream', 'Egg_pudding', 'Sweet_stewed_snow_pear', 'Fruit_salad', 'apple_pie', 'baby_back_ribs',
    'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad',
    'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake',
    'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',
    'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup',
    'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole',
    'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
    'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella',
    'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen',
    'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara',
    'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

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
        result, top5 = model_predict(np.array(img))
        result_json = {}
        for i in range(5):
            category = CLASSES[top5[-i - 1]]
            proba = '%.3f' % (result[0][top5[-i - 1]] * 100)   # 概率保留三位小数
            result_json[category] = str(proba)
        return result_json

    return None


if __name__ == '__main__':
    app.run()  # 云服务器上使用
    # 运行一个web服务器，端口为5000，作为本地测试使用
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # print('Check http://127.0.0.1:5000/')
    # http_server.serve_forever()
