import torch
import torch.nn as nn
from torch.nn.functional import embedding
from torchvision import models
import torchvision.models as models
from torch.functional import F  


class Encoder(nn.Module):
    def __init__(self, embed_size, pretrained=True, train_backbone=False, drop_prob=0.5) -> None:
        super(Encoder, self).__init__()
        self.pretrained = pretrained
        self.train_backbone = train_backbone
        # self.model = models.resnet152(pretrained=pretrained)
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_size)

        # modules = list(self.model.children())[:-1]
        # self.model = nn.Sequential(*modules)
        
        self.relu = nn.ReLU(drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.freeze()

    def forward(self, images):
        output = self.model(images)
        output = self.dropout(self.relu(output))
        return output

    def freeze(self):
        for name, param in self.model.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_backbone



class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, drop_prob=0.5) -> None:
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs



class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, pretrained=True, train_backbone=False, drop_prob=0.5) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(embed_size, pretrained, train_backbone, drop_prob)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        outputs = self.encoder(images)
        outputs = self.decoder(outputs, captions)
        return outputs

    def caption(self, image, vocabulary, max_length=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break

        return [vocabulary.itos[word_id] for word_id in result_caption]
