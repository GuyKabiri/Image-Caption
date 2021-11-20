import sys
from datetime import datetime
import torch
from torch.utils import data
from torch.utils.data import dataloader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import *
from dataloader import *
from model import *
from config import *


'''
    training and validation for one epoch
'''
def train_valid_one_epoch(model, loaders, writers, criterion, optimizer, scheduler, steps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kpi = {
        'train': {
            'loss': [],
        },
        'valid': {
            'loss': [],
        }
    }

    for phase in ['train', 'valid']:            #   iterate for training and validation phases
        if phase == 'train':
            model.train()
        elif loaders['valid']:
            model.eval()
        else:                   #   if validation loader is not defined
            continue

        # iterate over batches
        for idx, (images, captions) in tqdm(    
            enumerate(loaders[phase]), total=len(loaders[phase]), desc=phase
        ):
            max_length, batch_size, _ = captions.shape
            captions = captions.reshape(max_length, batch_size*5)

            images, captions = images.to(device), captions.to(device)           #   move itmes to gpu
            images = torch.repeat_interleave(images, repeats=5, dim=0) #[a, b, c] -> [aaaaa, bbbbb, ccccc].T

            if phase == 'train':
                outputs = model(images, captions[:-1])                          #   calculate outputs for training
            else:
                with torch.no_grad():
                    outputs = model(images, captions[:-1])                      #   calculate outputs for validation

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))   #   calculate loss

            kpi[phase]['loss'].append(loss.item())                              #   register this step loss

            if phase == 'train':
                writers['train'].add_scalar("steps loss", loss.item(), global_step=steps['train'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps[phase] += 1
            
            del images, captions, loss
            torch.cuda.empty_cache()
    
    return kpi, steps


'''
    training and validation for numerous of epochs
'''
def train_valid_epochs(model, loaders, writers, num_epochs, criterion, optimizer, scheduler, steps, run_path, prev_valid_loss=float('inf')):
    model_save_path = '{}/models/model_checkpoint.pth'.format(run_path)

    best_valid_loss = prev_valid_loss

    #   iterate over the epochs
    for epoch in range(1, num_epochs + 1):
        print('Epoch {:3d} of {}:'.format(epoch, num_epochs), flush=True)

        #   train and validation for one epoch
        kpi, steps = train_valid_one_epoch(model, loaders, writers, criterion, optimizer, scheduler, steps)

        #   pretty printing training and validation results
        print_str = ''
        for phase in ['train', 'valid']:
            if loaders[phase]:
                loss = sum(kpi[phase]['loss']) / len(loaders[phase])
                print_str += '{}:\tloss={:.5f}\n'.format(phase, loss)
                writers[phase].add_scalar('loss', loss, epoch)
        
        #   if validation loss is better, save model chechpoint
        epoch_loss = sum(kpi['train']['loss']) / len(loaders['train'])
        if epoch_loss < best_valid_loss:
            best_valid_loss = epoch_loss
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': steps['train'],
                'valid_step': steps['valid'],
                'epoch': epoch,
            }
            save_checkpoint(checkpoint, model_save_path)
        
        print_examples(model, loaders['train'].dataset)
        print(print_str)
    

'''
    train model by a predefined configuration
'''
def train():
    np.random.seed(42)
    torch.manual_seed(42)

    #   generate running environment
    datetime_srt = datetime.today().strftime("%d-%m-%y_%H:%M")
    run_path = os.path.join(sys.path[0], 'runs', datetime_srt)
    print('Generating running environment')
    create_env(run_path)
    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #   generate training and validation dataloaders
    print('Generating loaders')
    loaders = get_train_valid_loaders()

    #   update vocabulary size in config file
    CFG.vocab_size =  len(loaders['train'].dataset.vocab)
    CFG.save(run_path)

    #   generate training and validation writers
    print('Generating writers')
    writers = get_writers(run_path, CFG.model_name)
    steps = {
        'train': 0,
        'valid': 0
    }

    # initialize model, loss, optimizer, scheduler
    print('Generating model')
    model = EncoderDecoder(CFG.embed_size, CFG.hidden_size, CFG.vocab_size, CFG.lstm_num_layers, pretrained=CFG.pretrained, train_backbone=CFG.train_backbone, drop_prob=CFG.drop_rate).to(device)
    criterion = CFG.criterion(ignore_index=loaders['train'].dataset.vocab.stoi['<PAD>'])
    optimizer = CFG.optimizer(model.parameters(), **CFG.optimizer_dict)
    scheduler = CFG.scheduler(optimizer, **CFG.scheduler_dict) if CFG.scheduler else None

    if CFG.load_model:
        steps, epoch = load_checkpoint(torch.load(CFG.model_path), model, optimizer)

    train_valid_epochs(model, loaders, writers, CFG.num_epochs, criterion, optimizer, scheduler, steps, run_path)


if __name__ == "__main__":
    train()
