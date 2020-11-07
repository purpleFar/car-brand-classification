import os
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as trans
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from preprocessing import load_data, load_img, load_obj, device

Image.MAX_IMAGE_PIXELS = None


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential(
            [
                iaa.Resize((224, 224)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-20, 20), mode="symmetric"),
                iaa.Sometimes(
                    0.25,
                    iaa.OneOf(
                        [
                            iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(0.1, size_percent=0.5),
                        ]
                    ),
                ),
                iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            ]
        )

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def input_transform():
    return trans.Compose(
        [
            trans.Resize((224, 224)),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def my_transform():
    return trans.Compose(
        [
            ImgAugTransform(),
            lambda x: Image.fromarray(x),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class Car196Dataset(data.Dataset):
    def __init__(
        self, imgDir_label_list, input_transform=input_transform, is_train=False
    ):
        super(Car196Dataset, self).__init__()
        self.is_train = is_train
        [self.image_filenames, self.labels] = imgDir_label_list
        self.input_transform = input_transform

    def __getitem__(self, index):
        input_file = self.image_filenames[index]
        input_ = load_img(input_file)
        if self.input_transform:
            input_ = self.input_transform()(input_)
        return input_, self.labels[index]

    def __len__(self):
        return len(self.image_filenames)


def cut_CV_data(labels, k=5):
    class_num = {}
    class_list = {}
    k_fold = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        if label in class_num:
            class_num[label] += 1
            class_list[label].append(i)
        else:
            class_num[label] = 1
            class_list[label] = [i]
    for key, value in class_list.items():
        np.random.shuffle(class_list[key])
        tmp = 0
        for i in range(k - 1):
            k_fold[i] += class_list[key][tmp : tmp + class_num[key] // k]
            tmp += class_num[key] // k
        k_fold[-1] += class_list[key][tmp:]
    return k_fold


def train_early_stop(
    net,
    trainloader,
    valloader,
    optimizer,
    n_steps=800,
    p=10,
    savefile="./best_model.pt",
    show_acc=False,
    special_item=None,
    return_log=False,
    device=device,
):
    r"""

    Arguments:
    ---------
        net: (nn.Module)
            The model be trained.
        trainloader: (torch.utils.data.DataLoader)
            Torch dataloader in training.
        valloader: (torch.utils.data.DataLoader)
            ---
        optimizer:
            ---
        n_steps: (int)
            Get val accuracy and loss in each n_steps (default: ``100``).
        p: (int)
            Patience (default: ``6``).
        savefile: (string)
            The path of best model's parameters save (default: ``'./best_model.pt'``).
        show_acc: (boolean)
            Show valuation accuracy (default: ``False``).
        special_item: (function)
            f(model, inputs, labels)-->loss, outputs (default: ``None``).
        device: (string)
            'cpu' or 'cuda' (default: ``'cuda'``).
    """
    log = {"train": {"acc": [], "loss": []}, "val": {"acc": [], "loss": []}}
    prnt_loss_step = len(trainloader)
    if special_item is None:

        def input2lossAndOutputs(net, inputs, labels):
            outputs = net(inputs)
            return nn.CrossEntropyLoss()(outputs, labels), outputs

    else:
        input2lossAndOutputs = special_item

    def show_val_information(show_acc=True):
        net.eval()
        val_correct, val_loss, total_data = 0, 0, 0
        for i, (inputs, labels) in enumerate(valloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64)

            loss, outputs = input2lossAndOutputs(net, inputs, labels)
            val_loss += loss.item()
            total_data += int(inputs.shape[0])
            if show_acc:
                _, pred = outputs.max(1)
                val_correct += pred.eq(labels).sum().item()
        val_loss /= i + 1
        val_acc = 100.0 * val_correct / total_data
        print(
            "val accuracy: %.4f%% (%d/%d),  loss: %.3f"
            % (val_acc, val_correct, total_data, val_loss)
        )
        log["val"]["acc"].append(val_acc)
        log["val"]["loss"].append(val_loss)
        return val_acc, val_loss

    print("==> Training model..")
    best_val_loss = float("Infinity")
    if hasattr(optimizer, "optimizer"):
        scheduler = optimizer
        optimizer = scheduler.optimizer
    torch.save(net.state_dict(), savefile)
    epoch, best_steps, steps, j = 0, 0, 0, 0
    net.train()
    while True:  # loop
        epoch += 1
        running_loss = 0.0
        total_data, correct = 0, 0
        if "scheduler" in dir() and epoch != 1:
            scheduler.step()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            if steps % n_steps == 0:
                _, val_lost = show_val_information(show_acc=show_acc)
                net.train()
                if val_lost <= best_val_loss:
                    j = 0
                    torch.save(net.state_dict(), savefile)
                    best_steps = steps
                    best_val_loss = val_lost
                else:
                    j += 1
                    if j >= p:
                        print("Finished Training")
                        net.load_state_dict(torch.load(savefile))
                        best_val_acc, best_val_loss = show_val_information()
                        if return_log:
                            return net, best_steps, best_val_acc, best_val_loss, log
                        return net, best_steps, best_val_acc, best_val_loss
            total_data += int(inputs.shape[0])
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            loss, outputs = input2lossAndOutputs(net, inputs, labels)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            loss.backward()
            optimizer.step()
            steps += 1

            running_loss += loss.item()
            if (i + 1) % prnt_loss_step == 0:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch, i + 1, running_loss / (prnt_loss_step))
                )
                log["train"]["loss"].append(running_loss / (prnt_loss_step))
                running_loss = 0.0
        print(
            "%d epoch, training accuracy: %.4f%% (%d/%d)"
            % (epoch, 100.0 * correct / total_data, correct, total_data)
        )
        log["train"]["acc"].append(100.0 * correct / total_data)


def main():
    label = load_obj(os.path.join("preprocess_file", "label"))
    train_datapath = "training_data"

    [train_file, train_labels] = load_data(train_datapath, label, False)
    k_fold_num = 15
    k_fold = cut_CV_data(train_labels, k=k_fold_num)  # cross validation K-fold
    train_file = np.array(train_file)
    train_labels = np.array(train_labels)

    if not os.path.exists("model_wide_resnet"):
        os.makedirs("model_wide_resnet")
    for i in range(k_fold_num - 1):
        train_f, train_l = np.array([]), np.array([])
        for k in range(k_fold_num):
            if k != i:
                train_f = np.concatenate((train_f, train_file[k_fold[k]]), axis=0)
                train_l = np.concatenate((train_l, train_labels[k_fold[k]]), axis=0)
        train_dataset = Car196Dataset(
            [train_f, train_l], input_transform=my_transform, is_train=True
        )
        valid_dataset = Car196Dataset(
            [train_file[k_fold[i]], train_labels[k_fold[i]]], is_train=False
        )
        train_loader = DataLoader(
            train_dataset, num_workers=4, batch_size=16, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, num_workers=4, batch_size=16, shuffle=False
        )

        net = torch.hub.load(
            "pytorch/vision:v0.6.0", "wide_resnet50_2", pretrained=True
        )
        net.fc = nn.Linear(2048, 196)
        net = net.to(device)
        optimizer = optim.SGD(
            net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0045
        )
        stepLR = optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.8)

        train_early_stop(
            net,
            train_loader,
            valid_loader,
            stepLR,
            n_steps=1000,
            p=6,
            savefile=os.path.join("model_wide_resnet", "best_model{}.pt".format(i)),
            show_acc=True,
            return_log=True,
            device=device,
        )


if __name__ == "__main__":
    main()
