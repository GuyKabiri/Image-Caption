import torch
import torch.nn as nn
from torch.nn.functional import embedding
from torchvision import models
import torchvision.models as models
from torch.functional import F  


class Encoder(nn.Module):
    def __init__(self, embed_size, train_backbone=False) -> None:
        super(Encoder, self).__init__()
        self.train_backbone = train_backbone
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_backbone=False) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(embed_size, train_backbone)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        x = self.encoder(images)
        outputs = self.decoder(x, captions)
        return outputs

    def caption(self, image, vocabulary, max_length=50):
        result = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None
            
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result.append(predicted.item())
                x = self.decoder.embed(output).unsqueeze(0)
                
                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break
                
        return [vocabulary.itos[idx] for  idx in result]