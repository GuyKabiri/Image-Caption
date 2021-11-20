from __future__ import unicode_literals, print_function, division
from pandas.tseries import frequencies
from spacy import load
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from io import open
import glob
import os
import spacy
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import dataloader
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


spacy_eng = spacy.load('en_core_web_sm')


class Vocabulary:
    def __init__(self, freq_treshold) -> None: # freq_treshold - if a word doesnt appear enough times then it might not be important.
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"} # word that has frequence lower then freq treshhold will be mapped to UNKOWN
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_treshold = freq_treshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text): # seperating by space
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)] # Puts a sentence to a list. and DE-CAP the letters

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_treshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_treshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.images = os.listdir(self.root_dir)
        # self.imgs = self.df["image"]
        # self.captions = self.df["caption"]
        self.vocab = Vocabulary(freq_treshold)
        self.vocab.build_vocabulary(self.df.caption.tolist())        


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img_id = self.images[index]
        raw = self.df[self.df['image']==img_id]['caption'].tolist()
        caption = self.get_captions(raw)

        # caption = self.captions[index]
        # img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # numericalized_captions = []
        # for cap in captions:
        #     num_cap = [self.vocab.stoi["<SOS>"]] # stoi = string to index finding the start of the sentence
        #     num_cap += self.vocab.numericalize(cap) # do it for the rest of the caption
        #     num_cap.append(self.vocab.stoi["<EOS>"])
        #     numericalized_captions.append(torch.tensor(num_cap))


        return img, torch.as_tensor(caption)


    def get_captions(self, raw_cpations):
        numericalized_captions = [] 
        for caption in raw_cpations:
            num_cap = [self.vocab.stoi['<SOS>']] # stoi = string to index finding the start of the sentence
            num_cap += self.vocab.numericalize(caption) # do it for the rest of the caption
            num_cap.append(self.vocab.stoi['<EOS>'])
            numericalized_captions.append(torch.tensor(num_cap))
        captions = pad_sequence(numericalized_captions, batch_first=False, padding_value=self.vocab.stoi['<PAD>'])
        return captions


class MyCollate:
    def __init__(self, pad_idx) -> None:
        self.pad_idx = pad_idx

    def __call__(self, batch: torch.Tensor) -> Any:
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_transformer(phase):
    if phase == 'print':
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(400, 400)),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)),
            ])

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),
        ])

def get_train_valid_loaders(train_size=.75, batch_size=64, num_workers=4, shuffle=True, pin_memory=True, phase='train'):
    root_folder = "data/flickr8k/images/"
    annotation_file = "data/flickr8k/captions.txt"
    transform = get_transformer(phase)

    dataset = FlickerDataset(root_folder, annotation_file, transform=transform)

    indices = list(range(len(dataset)))
    split_train = int(np.floor(train_size * len(dataset)))
    split_valid = int(np.floor((1-train_size)/2 * len(dataset))) + split_train
    indices_dict = { 
        'train':    indices[ : split_train],
        'valid':    indices[ split_train : split_valid ],
        'test':     indices[ split_valid : ], 
    }

    if shuffle:
        samplers = { phase: SubsetRandomSampler(indices_dict[phase])
                            for phase in ['train', 'valid', 'test'] }
    else:
        samplers = { phase: Subset(dataset, indices_dict[phase])
                            for phase in ['train', 'valid', 'test'] }


    pad_idx = dataset.vocab.stoi["<PAD>"]

    loaders = {
        phase: DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx=pad_idx), sampler=samplers[phase])
                for phase in ['train', 'valid', 'test'] }

    return loaders