import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from dataloader import *


def print_examples(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transformer('test')

    model.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print("Example 1 OUTPUT: " + " ".join(model.caption(test_img1.to(device), dataset.vocab)))

    test_img2 = transform(Image.open("test_examples/child.jpg").convert("RGB")).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print("Example 2 OUTPUT: " + " ".join(model.caption(test_img2.to(device), dataset.vocab)))

    test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(0)
    print("Example 3 CORRECT: Bus driving by parked cars")
    print("Example 3 OUTPUT: " + " ".join(model.caption(test_img3.to(device), dataset.vocab)))

    test_img4 = transform(Image.open("test_examples/boat.png").convert("RGB")).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print("Example 4 OUTPUT: " + " ".join(model.caption(test_img4.to(device), dataset.vocab)))

    test_img5 = transform(Image.open("test_examples/horse.png").convert("RGB")).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print("Example 5 OUTPUT: " + " ".join(model.caption(test_img5.to(device), dataset.vocab)))


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_step = checkpoint["train_step"]
    valid_setp = checkpoint["valid_step"]
    steps = {
        'train': train_step,
        'valid': valid_setp
    }
    epoch = checkpoint["epoch"]
    return steps, epoch


def norm(img):
    img = np.array(img, dtype=np.float32)
    img -= img.min()
    img /= img.max()
    return img


def create_env(path):
    if not os.path.exists(path):
        os.mkdir(path)
    paths = ['logs', 'models', 'metrics']
    for p in paths:
        sub_path = os.path.join(path, p)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)


def get_writers(path, model_name, fold=None):
    if fold is not None:
        return { phase: SummaryWriter('{}/logs/{}_fold_{}_{}'.format(path,
                                                                    model_name,
                                                                    fold,
                                                                    phase))
                                                                            for phase in ['train', 'valid'] }

    return { phase: SummaryWriter('{}/logs/{}_{}'.format(path, model_name, phase))
                                                                                    for phase in ['train', 'valid'] }