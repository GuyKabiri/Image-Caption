import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from dataloader import get_loader
from model import *
from tqdm import tqdm


def train():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224))
            ])
            
    train_loader = get_loader(
        root_folder="data/flickr8k/images/",
        annotation_file="data/flickr8k/captions.txt",
        transform=transform
    )
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Hyperparameters 
    load_model = False
    embed_size = 256
    hidden_size = 256
    num_layers = 5
    num_epochs = 30
    learning_rate = 1e-5
    vocab_size = len(train_loader.dataset.vocab)


    # Tensorboard training monitor
    writer = SummaryWriter("logs/lab2")
    step = 0

    # initialize model, loss, optimizer etc.
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers)
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    save_model = True

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model = model.train().to(device)

    for epoch in range(num_epochs):
        # print_examples(model, device, dataset)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            score = model(imgs, captions[:-1])

            # print(score.shape, captions.shape)
            # print(score.reshape(-1, score.shape[2]).shape, captions.reshape(-1).shape)


            optimizer.zero_grad()
            loss = criterion(score.reshape(-1, score.shape[2]), captions.reshape(-1))
            # loss = criterion(score, captions)

            writer.add_scalar('loss', loss.item(), step)
            
            step += 1
            loss.backward()
            optimizer.step()
            del imgs, captions, loss
            torch.cuda.empty_cache()

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)


if __name__ == "__main__":
    train()
