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
def train_valid_one_epoch(model, loaders, writers, criterion, optimizer, steps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ngram = 4

    kpi = {

        phase: {
            'loss': [],
            'bleu': { '{}-gram'.format(i): 0
                                        for i in range(1, 1+ngram) }    #   generate dict for each n_gram
        }
        for phase in ['train', 'valid']
    }

    for phase in ['train', 'valid']:            #   iterate for training and validation phases
        if phase == 'train':
            model.train()
        elif loaders['valid']:
            model.eval()
        else:                   #   if validation loader is not defined
            continue

        # iterate over batches
        for idx, batch in tqdm(    
            enumerate(loaders[phase]), total=len(loaders[phase]), desc=phase
        ):
            (images, captions, _) = batch
            #   images shape    [batch, 3, 224, 224]
            #   captions shape  [max sentence length, batch, 5]
            max_length, batch_size, _ = captions.shape
            captions = captions.reshape(max_length, batch_size*5)

            images, captions = images.to(device), captions.to(device)           #   move itmes to gpu
            images = torch.repeat_interleave(images, repeats=5, dim=0)          #   [a, b, c] -> [aaaaa, bbbbb, ccccc].T

            #   images shape    [batch*5, 3, 224, 224]
            #   captions shape  [max sentence length, batch*5]
            with torch.set_grad_enabled(phase=='train'):
                outputs = model(images, captions[:-1])

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))   #   calculate loss
        
            kpi[phase]['loss'].append(loss.item())                              #   register this step loss
            writers[phase].add_scalar("steps loss", loss.item(), global_step=steps[phase])

            bleu_dict = bleu_score_(model, batch, loaders['train'].dataset)
            for n_gram in bleu_dict:
                kpi[phase]['bleu'][n_gram] += bleu_dict[n_gram]

            if phase == 'train':
                model.train()
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
def train_valid_epochs(model, loaders, writers, num_epochs, criterion, optimizer, scheduler, steps, run_path, prev_valid_loss=float('inf'), start_epoch=0):
    model_save_path = '{}/models/model_checkpoint.pth'.format(run_path)

    best_valid_loss = prev_valid_loss
    start_epoch += 1
    end_epoch = start_epoch + num_epochs

    #   iterate over the epochs
    for epoch in range(start_epoch, end_epoch):
        print('Epoch {:3d} of {}:'.format(epoch, end_epoch-1), flush=True)

        #   train and validation for one epoch
        kpi, steps = train_valid_one_epoch(model, loaders, writers, criterion, optimizer, steps) 

        #   pretty printing training and validation results
        print_str = ''
        for phase in ['train', 'valid']:
            if loaders[phase]:
                loss = sum(kpi[phase]['loss']) / len(loaders[phase])
                print_str += '{}:\tloss={:.5f}'.format(phase, loss)

                for n_gram in kpi[phase]['bleu']:
                    bleu = kpi[phase]['bleu'][n_gram] / len(loaders[phase])
                    print_str += '\t{}={:.5f}'.format(n_gram, bleu)
                    writers[phase].add_scalar(n_gram, bleu, epoch)

                print_str += '\n'
                writers[phase].add_scalar('loss', loss, epoch)

        epoch_val_loss = sum(kpi['valid']['loss']) / len(loaders['valid'])

        if scheduler is not None:
            try:
                lr = scheduler.get_last_lr()[0]
            except:
                lr = [ group['lr'] for group in optimizer.param_groups ][0]
            writers['train'].add_scalar('lr', lr, epoch)
            scheduler.step(epoch_val_loss)  

        #   if validation loss is better, save model chechpoint
        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
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


def test(model, loader, criterion, run_path):
    save_path = '{}/test/test.csv'.format(run_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ngram = 4

    df = pd.DataFrame(columns=['image', 'prediction', 'loss', *['{}-gram'.format(i) for i in range(1, 1+ngram)] ])

    model.eval()
    with torch.no_grad():
        # iterate over batches
        for idx, batch in tqdm(    
            enumerate(loader), total=len(loader), desc='testing'
        ):
            (image, caption, img_id) = batch
            #   images shape    [batch, 3, 224, 224]
            #   captions shape  [max sentence length, batch, 5]
            max_length, batch_size, _ = caption.shape
            caption = caption.reshape(max_length, batch_size*5)

            image, caption = image.to(device), caption.to(device)           #   move itmes to gpu

            words = ' '.join(model.caption(image, loader.dataset.vocab))    #   caption before multiple the images to 5 each one

            image = torch.repeat_interleave(image, repeats=5, dim=0)          #   [a, b, c] -> [aaaaa, bbbbb, ccccc].T

            #   images shape    [batch*5, 3, 224, 224]
            #   captions shape  [max sentence length, batch*5]
            outputs = model(image, caption[:-1])

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), caption.reshape(-1)).cpu().detach().item()   #   calculate loss
            bleu_dict = bleu_score_(model, batch, loader.dataset)

            df = df.append({ 'image': img_id[0], 'prediction': words, 'loss': loss, **bleu_dict }, ignore_index=True)

            del image, caption, loss, words
            torch.cuda.empty_cache()

    df. to_csv(save_path)

    

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
    loaders = get_loaders(train_size=CFG.train_size, batch_size=CFG.batch_size)

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
    criterion = CFG.criterion(ignore_index=loaders['train'].dataset.vocab.stoi['<PAD>'], **CFG.criterion_dict)
    optimizer = CFG.optimizer(model.parameters(), **CFG.optimizer_dict)
    scheduler = CFG.scheduler(optimizer, **CFG.scheduler_dict) if CFG.scheduler else None
    start_epoch = 0

    if CFG.load_model:
        steps, start_epoch = load_checkpoint(torch.load(CFG.model_path), model, optimizer)

    train_valid_epochs(model, loaders, writers, CFG.num_epochs, criterion, optimizer, scheduler, steps, run_path, start_epoch=start_epoch)
    test(model, loaders['test'], criterion, run_path)




if __name__ == "__main__":
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    train()
