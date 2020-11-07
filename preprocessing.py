from os.path import basename
import os
import glob
import numpy as np
import csv
import pickle
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use", device)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    return img


def load_data(image_dir, label=None, clean=True):
    image_filenames, labels = [], []
    path_pattern = image_dir + "/**/*.*"
    files_list = glob.glob(path_pattern, recursive=True)
    for file in files_list:
        if is_image_file(file):
            image_filenames.append(file)
            if clean:
                img = np.array(load_img(file))
                if len(img.shape) < 3:
                    imgArray = np.repeat(
                        img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2
                    )
                    im = Image.fromarray(imgArray)
                    im.save(file)
            element = -1
            if not label is None:
                element = label[int(basename(file).split(".")[0])]
            labels.append(element)
    return [image_filenames, labels]


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def main():
    num2name, name2num = {}, {}
    label = {}
    with open("training_labels.csv", newline="") as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        n = 0
        for row in rows:
            if not row[1] in name2num:
                num2name[n] = row[1]
                name2num[row[1]] = n
                n += 1
            label[int(row[0])] = name2num[row[1]]

    if not os.path.exists("preprocess_file"):
        os.makedirs("preprocess_file")
    save_obj(num2name, os.path.join("preprocess_file", "num_to_name"))
    save_obj(name2num, os.path.join("preprocess_file", "name_to_num"))
    save_obj(label, os.path.join("preprocess_file", "label"))

    train_datapath = "training_data"
    test_datapath = "testing_data"
    load_data(train_datapath, clean=True)
    load_data(test_datapath, clean=True)


if __name__ == "__main__":
    print("Preprocessing the data...", end="")
    main()
    print("done")
