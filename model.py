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
        self.model = None
        self.model.fc = None
        self.freeze()

    def forward(self, images):
        """
        The Encoder takes in images and basicly extracts the features with the backbone.
        Write the process in code below.
        """
        return output

    def freeze(self):
        for name, param in self.model.named_parameters:
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_backbone


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = None
        self.linear = None

    def forward(self, features, captions):
        """
        Use the skeleton above and code the Decoder forward-pass
        **features are the extracted features we obtained from the bacbone.**
        Fill me!
        """
        return outputs


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train_backbone=False) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, images, captions):
        x = self.encoder(images)
        outputs = self.decoder(x, captions)
        return outputs

    def caption(self, image, vocabulary, max_length=50):
        """
        Our image caption inference!
        take in an image and vocabulary
        run the model on the image and output the caption!
        """
        result = []
        return [vocabulary.itos[idx] for  idx in result]