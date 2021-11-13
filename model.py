import torch
import torch.nn as nn
from torch.nn.functional import embedding
from torchvision import models
import torchvision.models as models
from torch.functional import F  


class Encoder(nn.Module):
    def __init__(self, embed_size, train_backbone=False, drop_prob=0.3) -> None:
        super(Encoder, self).__init__()
        self.train_backbone = train_backbone
        resnet = models.resnet50(pretrained=True)

        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules)
        
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.3)
        self.freeze()

        # self.dropout = nn.Dropout(0.3)
        # self.relu = nn.ReLU()

    def forward(self, images):
        output = self.model(images)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc(output)
        return output

    def freeze(self):
        for name, param in self.model.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_backbone



class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, drop_prob=0.3) -> None:
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        print(features.shape, captions.shape)
        embeds = self.embed(captions[:-1])
        print(features.unsqueeze(0).shape, embeds.shape)
        x = torch.cat((features.unsqueeze(0), embeds), dim=0) 
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x



class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_backbone=False, drop_prob=0.3) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(embed_size, train_backbone, drop_prob)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        outputs = self.encoder(images)
        outputs = self.decoder(outputs, captions)
        return outputs

    def caption(self, image, vocabulary, max_length=50):
        captions = []
        batch_size = image.size(0)

        with torch.no_grad():            
            for i in range(max_length):
                output, hidden = self.decoder.lstm(inputs, hidden)
                output = self.decoder.linear(output)
                output = output.view(batch_size, -1)
            
                #select the word with most val
                predicted_word_idx = output.argmax(dim=1)
                
                #save the generated word
                captions.append(predicted_word_idx.item())
                
                #end if <EOS detected>
                if vocabulary.itos[predicted_word_idx.item()] == '<EOS>':
                    break
                
                #send generated word as the next caption
                inputs = self.decoder.embed(predicted_word_idx.unsqueeze(0))
            
            #covert the vocab idx to words and return sentence
            return [vocabulary.itos[idx] for idx in captions]
        #     x = self.encoder(image).unsqueeze(0)
        #     states = None
            
        #     for _ in range(max_length):
        #         hiddens, states = self.decoder.lstm(x, states)
        #         output = self.decoder.linear(hiddens.squeeze(0))
        #         predicted = output.argmax(1)
        #         result.append(predicted.item())
        #         x = self.decoder.embed(output).unsqueeze(0)
                
        #         if vocabulary.itos[predicted.item()] == '<EOS>':
        #             break
                
        # return [vocabulary.itos[idx] for  idx in result]
