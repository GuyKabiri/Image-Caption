import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from .utils import save_checkpoint, load_checkpoint, print_examples
from .dataloader import get_loader
from .model import EncoderDecoder


def train():
    train_loader, dataset = get_loader(
        root_folder="data/flickr8k/images",
        annotation_file="data/flickr8k/captions.txt",
    )
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Hyperparameters 


    # Tensorboard training monitor
    writer = SummaryWriter("logs/lab3")
    step = 0

    # initialize model, loss, optimizer etc.
    model = None 
    criterion = None 
    optimizer = None
    save_model = True

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        # print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            pass



if __name__ == "__main__":
    train()
