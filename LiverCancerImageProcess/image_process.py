import os
import pickle
import numpy as np


def gen_dict(root_dir):
    d = {"labels": [], "data": np.array([], dtype=np.uint8), "filenames": []}

    def scan_dir(label, dir):
        for file in os.listdir(dir):
            d["labels"].append(label)
            with open(dir + '/' + file, "rb") as image:
                b = bytearray(image.read())
                if b:
                    d["data"] = np.append(d["data"], b[0])
            d["filenames"].append(file)

    for name in os.listdir(root_dir):
        full_name = root_dir + '/' + name
        if os.path.isdir(full_name):
            if name == "Normal_Processed":
                scan_dir(0, full_name)
            elif name == "Rat_HCC_HE_Processed":
                scan_dir(1, full_name)
            else:
                continue

    return d


def dump_dict(root_dir):
    d = gen_dict(root_dir)
    with open('./batch.pickle', 'wb') as file:
        pickle.dump(d, file)


def unpickle_dict(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


dump_dict("/media/cjiang/Extreme SSD")

#{b'labels': [], b'data': array([[], [], ..., [], []], dtype=uint8), b'filenames': []}