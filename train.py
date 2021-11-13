import torch
from torch.utils import data
from torch.utils.data import dataloader
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
    print(device)


    # Hyperparameters 
    load_model = False
    embed_size = 400
    hidden_size = 512
    vocab_size =  len(train_loader.dataset.vocab)
    num_layers = 5
    learning_rate = 1e-4
    num_epochs = 2
    print_every = 100


    # Tensorboard training monitor
    writer = SummaryWriter("logs/lab2")
    step = 0

    # initialize model, loss, optimizer etc.
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    save_model = True

    if load_model:
        step = load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

    model = model.train()
    
    for epoch in range(1, num_epochs + 1):   
        for idx, (images, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            images, captions = images.to(device), captions.to(device)

            optimizer.zero_grad()

            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            loss.backward()
            optimizer.step()
            
            if (idx + 1) % print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))
                
                
                # #generate the caption
                # model.eval()
                # with torch.no_grad():
                #     dataiter = iter(dataloader)
                #     img,_ = next(dataiter)
                #     features = model.encoder(img[0:1].to(device))
                #     caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=dataset.vocab)
                #     caption = ' '.join(caps)
                #     print_examples(img[0], title=caption,)
                    
                # model.train()
            
            del images, captions, loss
            torch.cuda.empty_cache()












    # for epoch in range(num_epochs):
    #     # print_examples(model, device, dataset)

    #     for idx, (imgs, captions) in tqdm(
    #         enumerate(train_loader), total=len(train_loader)
    #     ):
    #         imgs = imgs.to(device)
    #         captions = captions.to(device)
            
    #         score = model(imgs, captions[:-1])

    #         # print(score.shape, captions.shape)
    #         # print(score.reshape(-1, score.shape[2]).shape, captions.reshape(-1).shape)


    #         optimizer.zero_grad()
    #         loss = criterion(score.reshape(-1, score.shape[2]), captions.reshape(-1))
    #         # loss = criterion(score, captions)

    #         writer.add_scalar('loss', loss.item(), step)
            
    #         step += 1
    #         loss.backward()
    #         optimizer.step()
    #         del imgs, captions, loss
    #         torch.cuda.empty_cache()

    #     if save_model:
    #         checkpoint = {
    #             "state_dict": model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "step": step,
    #         }
    #         save_checkpoint(checkpoint)


if __name__ == "__main__":
    train()
