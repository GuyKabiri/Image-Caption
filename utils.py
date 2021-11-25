import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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
    paths = ['logs', 'models', 'test']
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


def clean_sentence(sentence):
    for i, item in enumerate(sentence):
        if item == '<SOS>':
            sentence.remove(item)

        if item == '<EOS>':
            del sentence[i: len(sentence)] # remove <EOS> and <PAD> after it
    return sentence


def bleu_score_(model, batch, dataset, n=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_grams = { '{}-gram'.format(i): []
                                        for i in range(1, 1+n) }    #   generate dict for each n_gram

    images, captions, _ = batch
    smoothie = SmoothingFunction().method1

    with torch.no_grad():
        for i, img in enumerate(images):    #   iterate over images
            img = img.unsqueeze(0)          #   remove the batch axis

            #   generate text sentence
            captions_list = [clean_sentence([dataset.vocab.itos[idx.detach().cpu().item()] for idx in captions[:, i, j]]) for j in range(5)]
            output = clean_sentence(model.caption(img.to(device), dataset.vocab))

            #   calculate n_grams for each images
            for i in range(1, 1+n):
                w = 1/i                 #   weight to assign
                weights = tuple(w if j<i else 0 for j in range(n))  #   generate vector of weights e.g.: [0.33, 0.33, 0.33, 0] for 3-gram
                name = '{}-gram'.format(i)
                gram = sentence_bleu(captions_list, output, weights=weights, smoothing_function=smoothie)
                n_grams[name] = np.mean(np.array(gram))

    return n_grams