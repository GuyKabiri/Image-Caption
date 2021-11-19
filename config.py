import torch.nn as nn
import torch.optim as optim

class CFG:
    model_name = 'resnet152'
    model_path = None
    load_model = False
    embed_size = 512
    hidden_size = 512
    vocab_size =  2994
    lstm_num_layers = 2
    learning_rate = 1e-3
    num_epochs = 150
    drop_rate = 0.5
    pretrained = True
    train_backbone = False

    criterion = nn.CrossEntropyLoss

    optimizer = optim.Adam
    optimizer_dict = {  'lr': learning_rate, }

    scheduler = None
    scheduler_dict = None
    
    # scheduler = optim.lr_scheduler.StepLR
    # scheduler_dict = { 'step_size': 5,
    #                     'gamma':    0.1 }

    def save(path):
        save_path = path + '/model_dict.json'
        with open(save_path, 'w') as f:
            for key, val in CFG.__dict__.items():
                f.write('{}\t\t= {}\n'.format(key, val))