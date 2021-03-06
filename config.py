import torch.nn as nn
import torch.optim as optim

class CFG:
    seed = 42
    model_name = 'inceptionv3'
    model_path = 'runs/26-11-21_20:20/models/model_checkpoint.pth'
    load_model = True
    embed_size = 512
    hidden_size = 512
    vocab_size =  2994
    lstm_num_layers = 3
    batch_size = 64
    learning_rate = 1e-5
    num_epochs = 50
    drop_rate = 0.6
    train_size = 0.75
    pretrained = True
    train_backbone = False

    criterion = nn.CrossEntropyLoss
    criterion_dict = { }

    optimizer = optim.Adam
    optimizer_dict = {  'lr': learning_rate, }

    scheduler = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_dict = { 'factor': 0.8, 'patience': 2 }

    def save(path):
        save_path = path + '/model_dict.json'
        with open(save_path, 'w') as f:
            for key, val in CFG.__dict__.items():
                f.write('{}\t\t= {}\n'.format(key, val))