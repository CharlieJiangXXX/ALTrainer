import os
import pickle
import numpy as np
from tqdm import tqdm


def gen_dict(root_dir) -> dict:
    out_dict = {"labels": [], "data": np.array([], dtype=np.uint8), "filenames": []}

    def scan_dir(label, dir):
        for file in tqdm(os.listdir(dir)):
            out_dict["labels"].append(label)
            with open(dir + '/' + file, "rb") as image:
                b = bytearray(image.read())
                if b:
                    out_dict["data"] = np.append(out_dict["data"], b)
            out_dict["filenames"].append(file)

    for name in os.listdir(root_dir):
        full_name = root_dir + '/' + name
        if os.path.isdir(full_name):
            if name == "Normal_Processed":
                print("Processing normal...")
                scan_dir(0, full_name)
            elif name == "Rat_HCC_HE_Processed":
                print("Processing rat HCC HE...")
                scan_dir(1, full_name)
            continue
    return out_dict


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