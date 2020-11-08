from os.path import basename
import os
import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
from preprocessing import load_data, load_obj, device
from train import Car196Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dir",
        help="the directory of the testing images",
        default="testing_data",
        type=str,
    )
    parser.add_argument(
        "--model_dir", help="the directory of models", default="model", type=str
    )
    parser.add_argument(
        "--save_name", help="the name of submission", default="submission.csv", type=str
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_datapath = args.test_dir
    model_dir = args.model_dir
    save_name = args.save_name

    num2name = load_obj(os.path.join("preprocess_file", "num_to_name"))
    test_dataset = Car196Dataset(load_data(test_datapath, clean=False), is_train=False)

    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=16, shuffle=False)

    net = torch.hub.load("pytorch/vision:v0.6.0", "wide_resnet50_2", pretrained=False)
    net.fc = nn.Linear(2048, 196)
    net = net.to(device)

    path_pattern = model_dir + "/**/*.*"
    files_list = glob.glob(path_pattern, recursive=True)

    csv_list = [["id", "label"]]
    tmp = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = torch.zeros(
            inputs.shape[0], 196, dtype=torch.float64, device=device
        ).data
        for file_name in files_list:
            net.load_state_dict(torch.load(file_name))
            net.eval()
            outputs += nn.Softmax(dim=1)(net(inputs)).data
        _, preds = outputs.max(1)
        for index, pred in enumerate(preds):
            input_file = test_dataset.image_filenames[tmp + index]
            id_ = int(basename(input_file).split(".")[0])
            csv_list.append([id_, num2name[pred.item()]])
        tmp += index + 1

    if not os.path.exists("result"):
        os.makedirs("result")
    with open(os.path.join("result", save_name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)


if __name__ == "__main__":
    print("Processing...")
    main()
    print("done")
