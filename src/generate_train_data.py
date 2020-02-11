import numpy as np
import time
from haifu_parser import load_data
from haifu_parser import richi_filter
from haifu_parser import parse_haifu
from haifu_parser import action_to_vector


def generate_train_data(file_name):
    # 600 haifus/1s
    print("Hiafu name:" + file_name)
    time_start = time.time()
    x_data = []
    y_data = []

    file_path = "../data/" + file_name + ".txt"
    # test_list = load_data("../data/sample.txt")
    # test_list = load_data("../data/totuhaihu.txt")
    test_list = load_data(file_path)

    new_richi_data = richi_filter(test_list)

    print("Embed to vectors:")
    for haifu in new_richi_data:
        inp, chanfon, jikaze, dora_list, tenpai_result, sute = parse_haifu(haifu)
        for each_inp in inp:
            x = []
            player = each_inp[0]
            for action in each_inp.split(" "):
                if action != "":
                    if not (player != action[0] and action[1] == "G"):
                        x.append(action_to_vector(action, player, chanfon, jikaze, dora_list))
            x_data.append(np.array(x))
            y_data.append(tenpai_result)
            if len(y_data) % 2000 == 0:
                print(len(y_data), round(time.time() - time_start, 2))
    x_data_numpy = np.array(x_data)
    y_data_numpy = np.array(y_data)

    time_end = time.time()
    print('Generate train data cost %s seconds.' %
          round((time_end - time_start), 2))

    return x_data_numpy, y_data_numpy

def pad_x(x_data):
    x_len = []
    for i in range(len(x_data)):
        x_len.append(x_data[i].shape[0])
    max_x_len = max(x_len)
    x_data_ret = np.zeros((len(x_data), max_x_len, 52))
    
    for i in range(len(x_data)):
        zeros = np.zeros((max_x_len - x_data[i].shape[0], 52))
        x_data_ret[i] = np.concatenate((zeros, x_data[i]), axis=0)
    return x_data_ret


local_place = "/home/lius/Tenpai_prediction-master/data/"
def save_train_data():
    x_data, y_data= generate_train_data("mjscore")
    x_data = x_data[:50000]
    y_data = y_data[:50000]
    x_data = pad_x(x_data)
    np.save(local_place + "x_data.npy", x_data[:50000])
    np.save(local_place + "y_data.npy", y_data[:50000])

def generate_train_test_local():
    x_data = np.load(local_place + "x_data.npy")
    y_data = np.load(local_place + "y_data.npy")
    return x_data,y_data
