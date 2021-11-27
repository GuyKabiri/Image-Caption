# Image Captioning

The following notebook is an exercise for the Convolutional Neural Networks for Computer Vision course at Afeka College of Engineering.  
It uses Flickr8K dataset for image captioning.

Submitted By:  
*   Tal Goldengoren  
*   Guy Kabiri

Table of Contents:
- [Image Captioning](#image-captioning)
  - [Imports](#imports)
  - [Data Exploration](#data-exploration)
    - [Understand the Data](#understand-the-data)
    - [Data Processing](#data-processing)
      - [Image Processing](#image-processing)
      - [Captions Processing](#captions-processing)
    - [Data Samples](#data-samples)
  - [Training](#training)
    - [Training Process](#training-process)
  - [Graphs](#graphs)
    - [1-gram](#1-gram)
    - [2-gram](#2-gram)
    - [3-gram](#3-gram)
    - [4-gram](#4-gram)
    - [Loss](#loss)
    - [Learning Rate](#learning-rate)
  - [Prediction Results](#prediction-results)
    - [Good Predictions](#good-predictions)
    - [Bad Predictions](#bad-predictions)
  - [Refereneces](#refereneces)

## Imports


```python
from dataloader import *
from model import *
from train import *
from utils import *

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
```


```python
assert torch.cuda.is_available()
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed(CFG.seed)
```

## Data Exploration

The dataset used in this exercise was Flickr8K.  
It contains about 8,000 images, with 5 different captions each. Therefore, a total of about 40,000 captions.  
As each image may be described in different ways by different people, having more than 1 caption for each image will assist in better training and evaluating the correctness of the predictions.


```python
loader = get_loaders(batch_size=1, phase='test')['test']

captions_file = "data/flickr8k/captions.txt"
df = pd.read_csv(captions_file)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 40455 entries, 0 to 40454
    Data columns (total 2 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   image    40455 non-null  object
     1   caption  40455 non-null  object
    dtypes: object(2)
    memory usage: 632.2+ KB


Some captions examples


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>caption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A child in a pink dress is climbing up a set o...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A girl going into a wooden building .</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A little girl climbing into a wooden playhouse .</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A little girl climbing the stairs to her playh...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000268201_693b08cb0e.jpg</td>
      <td>A little girl in a pink dress going into a woo...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>40450</th>
      <td>997722733_0cb5439472.jpg</td>
      <td>A man in a pink shirt climbs a rock face</td>
    </tr>
    <tr>
      <th>40451</th>
      <td>997722733_0cb5439472.jpg</td>
      <td>A man is rock climbing high in the air .</td>
    </tr>
    <tr>
      <th>40452</th>
      <td>997722733_0cb5439472.jpg</td>
      <td>A person in a red shirt climbing up a rock fac...</td>
    </tr>
    <tr>
      <th>40453</th>
      <td>997722733_0cb5439472.jpg</td>
      <td>A rock climber in a red shirt .</td>
    </tr>
    <tr>
      <th>40454</th>
      <td>997722733_0cb5439472.jpg</td>
      <td>A rock climber practices on a rock climbing wa...</td>
    </tr>
  </tbody>
</table>
<p>40455 rows × 2 columns</p>
</div>



As can be seen above, each image has 5 captions, it means that during training all 5 captions should be taken into account when evaluating models performance.


```python
num_images = len(df.image.unique())
train_img_size, valid_img_size, test_img_size = int(num_images*CFG.train_size), int(num_images*(1-CFG.train_size)/2), int(num_images*(1-CFG.train_size)/2)
train_cap_size, valid_cap_size, test_cap_size = train_img_size*5, valid_img_size*5, test_img_size*5
print('There are {} images in the dataset'.format(num_images))
print('Training set will contain {} images and {} captions'.format(train_img_size, train_cap_size))
print('Validation set will contain {} images and {} captions'.format(valid_img_size, valid_cap_size))
print('Test set will contain {} images and {} captions'.format(test_img_size, test_cap_size))
```

    There are 8091 images in the dataset
    Training set will contain 6068 images and 30340 captions
    Validation set will contain 1011 images and 5055 captions
    Test set will contain 1011 images and 5055 captions


### Understand the Data


```python
loader_iter = iter(loader)
_, caps, _ = next(loader_iter)
print(caps)
```

    tensor([[[  1,   1,   1,   1,   1]],
    
            [[  4,   4,  10,  10, 431]],
    
            [[ 30, 431, 431,  21, 335]],
    
            [[  6,   6,  30,   6,   6]],
    
            [[ 29,  17,   6,  17,  29]],
    
            [[ 37, 324,  17,  29,  37]],
    
            [[ 10,  37,  29,   8,  44]],
    
            [[ 44, 423,   8,  10,   2]],
    
            [[  5,  44,  10, 423,   0]],
    
            [[  2,   5,  44,  44,   0]],
    
            [[  0,   2,   5,   5,   0]],
    
            [[  0,   0,   2,   2,   0]]])



```python
for _ in range(2):
    batch = next(loader_iter)
    imgs, caps, _ = batch
    print('Images shape: {}'.format(imgs.shape))
    print('Captions shape: {}'.format(caps.shape))
    print()
```

    Images shape: torch.Size([1, 3, 224, 224])
    Captions shape: torch.Size([16, 1, 5])
    
    Images shape: torch.Size([1, 3, 224, 224])
    Captions shape: torch.Size([21, 1, 5])
    


The data will be provided to the model as follow:  
Images:     [B, C, H, W]  
Captions:   [MS, B, NC]

B=batch size  
MS=max sentence length  
NC=number of captions per image

As the images shape is quite understandable, the captions is a bit weird.  
This shape is due to the different sentences length between the different samples.  
When working with batches, the samples whitin each batch should be equals size, therefore, it is not possible to represent sentences with different lengths with a normal shape, and much easier to padding short sentences in that shape.  
The first sentence present along the first column of the matrix, the second sentence in the second column, and so on.

### Data Processing


#### Image Processing
The images in the dataset are variety in shapes.  
The backbone model which will be used in this architecture, will be a pre-trained model (ImageNet), therefore all the images will be resized into 224X244 shape.  
Also, because the model is pre-trained, the images will be normalized into ImageNet mean and std values.

#### Captions Processing
As nueral networks understand only numbers, and not words, all of the captions need to be transformed into numbers.  
It means that each unique word in the dataset should get a unique number to reprenet it.  
For this task, a pre-build vocabulary will be used, this vocabulary contains a large amount of words, each will be mapped into a unique number.  
As dataset may contains words that appear only once in captions, the model will have hard time learning such words.  
Therefore, only frequent words will be taking into account, while leaving the un-common words out, this can be addjust by a threshold, which means it is another hyper-parameter that can be tuned.  
Moreover, the tokkenized vocabulary will hold a several unique words that have a special meaning:
*   `<SOS>` - Start of sentence
*   `<EOS>` - End of sentence
*   `<PAD>` - Paddind to generate equal size captions during training
*   `<UKN>` - Any word under the frequent threshold

### Data Samples


```python
def get_sentences(dataloader, captions):
    captions = captions.squeeze(1)
    captions = torch.permute(captions, (1, 0))
    num_sentences, num_word = captions.shape
    sentences = []
    for i in range(num_sentences):
        words = [ dataloader.dataset.vocab.itos[int(word)] for word in captions[i] ]    #   convert tokenizes to words
        eos_index = words.index('<EOS>')        #   find index of <EOS>
        words = words[1 : eos_index]            #   remove <SOS> and <EOS>
        sen = ' '.join(words)
        sentences.append(sen)

    return sentences
```


```python
def show_example(dataloader, rows=4, cols=2):
    num_examples = cols*rows
    global_offset = 14
    font_size = 12
    transform = get_transformer('print')


    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(10*cols, 10*rows))
    for idx, (_, captions, img_id) in enumerate(dataloader):
        if idx == num_examples:
            break

        img = transform(Image.open('data/flickr8k/images/' + img_id[0]).convert('RGB'))

        # img = img.squeeze(0)
        img = np.transpose(img, (1, 2, 0))
        sentences = get_sentences(dataloader, captions)

        ridx, cidx = idx//cols, idx%cols
        axs[ridx, cidx].imshow(norm(img))
        offset = global_offset
        for sen in sentences:
            axs[ridx, cidx].text(2, offset, sen, fontsize=font_size, color='white', bbox=dict(facecolor='black', alpha=0.5))
            offset += global_offset
        axs[ridx, cidx].axis('off')
    plt.tight_layout()
    plt.show()

show_example(loader)
```


    
![1](https://user-images.githubusercontent.com/52006798/143674620-470e0c16-9d87-43c9-b634-7d69fbcf0081.png)
    


## Training
The training process involved several configuration and trials:  
Two backbone modleds were tested for the encoder, Resnet-152 and InceptionV3.  
Various amount of LSTM layers were tested from 2, up to 5.  
Several learning rates, as well as, different number of epochs and batch sizes.  

For final configuration the following was used:
*   Backbone: InceptionV3
*   Embedded Size: 512
*   Hidden Size: 512
*   LSTM Layers: 3
*   Batch Size: 32
*   learning_rate: 1e-4
*   num_epochs: 150
*   drop_rate: 0.5
*   Criterion: CrossEntropyLoss
*   Optimizer: Adam
*   Scheduler: ReduceLROnPlateau w/ factor=0.8, patience=2

The backbone was a pre-trained model, and it was not trained during the training phase.  

### Training Process

![](https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/03-advanced/image_captioning/png/model.png)  

During training, first, an image goes through the CNN model in order to extract its features.  
After extracting features, a linear layer will be used to map the features into the vocabulary embedding size, with a dropout layer on top of it for bettrer training.  
Later on, this linear layer inserted into the decoder, which will pass the output of the embedding layer into certain amount of LSTM layers, in order to generate sequence of words.  
For final prediction, a linear layer with the size of the vucabulary will be used to map the prediction to the correct words.

## Graphs

### 1-gram
![1-gram](https://user-images.githubusercontent.com/52006798/143674081-328926cf-eba7-4ea4-a326-b4583c40b102.png)

### 2-gram
![2-gram](https://user-images.githubusercontent.com/52006798/143674083-df26773c-674b-4d7d-95e7-6087f3b4da40.png)

### 3-gram
![3-gram](https://user-images.githubusercontent.com/52006798/143674084-3c4498f6-97b7-4331-87dd-2c9fb4219144.png)

### 4-gram
![4-gram](https://user-images.githubusercontent.com/52006798/143674085-6570390a-02d0-4c03-8a28-a1b8f6a32b71.png)

### Loss
![loss](https://user-images.githubusercontent.com/52006798/143674086-7ce0da65-1e2b-48d7-aa34-5c61d8f3a9f2.png)

### Learning Rate
![lr](https://user-images.githubusercontent.com/52006798/143674087-c3b51490-6c14-4c04-8f17-4f9ec3b8c058.png)

## Prediction Results


```python
# test_path = 'runs/26-11-21_10:36/test'
# test_path = 'runs/26-11-21_16:03/test'
test_path = 'runs/26-11-21_20:20/test'
test_df = pd.read_csv(test_path + '/test.csv')
```


```python
test_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>image</th>
      <th>prediction</th>
      <th>loss</th>
      <th>1-gram</th>
      <th>2-gram</th>
      <th>3-gram</th>
      <th>4-gram</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3150659152_2ace03690b.jpg</td>
      <td>&lt;SOS&gt; a man is standing on a rock overlooking ...</td>
      <td>3.138403</td>
      <td>0.636364</td>
      <td>0.356753</td>
      <td>0.112244</td>
      <td>0.064841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2222498879_9e82a100ab.jpg</td>
      <td>&lt;SOS&gt; a dog is jumping over a hurdle . &lt;EOS&gt;</td>
      <td>1.556955</td>
      <td>0.625000</td>
      <td>0.422577</td>
      <td>0.143842</td>
      <td>0.087836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3126752627_dc2d6674da.jpg</td>
      <td>&lt;SOS&gt; a football player in a red uniform is ru...</td>
      <td>1.948640</td>
      <td>0.427367</td>
      <td>0.181596</td>
      <td>0.065234</td>
      <td>0.040041</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3257207516_9d2bc0ea04.jpg</td>
      <td>&lt;SOS&gt; a man in a black shirt and a woman in a ...</td>
      <td>3.116272</td>
      <td>0.357143</td>
      <td>0.230022</td>
      <td>0.182766</td>
      <td>0.125008</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2289096282_4ef120f189.jpg</td>
      <td>&lt;SOS&gt; a man and a woman are sitting on a bench...</td>
      <td>3.108447</td>
      <td>0.411765</td>
      <td>0.160422</td>
      <td>0.055566</td>
      <td>0.033272</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>1007</td>
      <td>1303727066_23d0f6ed43.jpg</td>
      <td>&lt;SOS&gt; a man in a black shirt and a woman in a ...</td>
      <td>3.242519</td>
      <td>0.230769</td>
      <td>0.096077</td>
      <td>0.033755</td>
      <td>0.020222</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>1008</td>
      <td>534886684_a6c9f40fa1.jpg</td>
      <td>&lt;SOS&gt; a man in a black shirt and jeans is stan...</td>
      <td>2.602398</td>
      <td>0.529412</td>
      <td>0.363803</td>
      <td>0.095914</td>
      <td>0.050105</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>1009</td>
      <td>2431723485_bc6b8e6418.jpg</td>
      <td>&lt;SOS&gt; a man in a red shirt and a black dog are...</td>
      <td>2.363976</td>
      <td>0.394458</td>
      <td>0.203299</td>
      <td>0.061354</td>
      <td>0.034292</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>1010</td>
      <td>3373481779_511937e09d.jpg</td>
      <td>&lt;SOS&gt; a man in a red shirt and white shorts is...</td>
      <td>2.990329</td>
      <td>0.500000</td>
      <td>0.196116</td>
      <td>0.068436</td>
      <td>0.041316</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>1011</td>
      <td>3265964840_5374ed9c53.jpg</td>
      <td>&lt;SOS&gt; a man in a red jacket is riding a bike o...</td>
      <td>1.752711</td>
      <td>0.600000</td>
      <td>0.462910</td>
      <td>0.320647</td>
      <td>0.228942</td>
    </tr>
  </tbody>
</table>
<p>1012 rows × 8 columns</p>
</div>




```python
def get_clean_sentence(sentence):
    stopwords = ['<SOS>', '<EOS>', '.']
    words_list = sentence.split()
    resultwords = [word for word in words_list if word.upper() not in stopwords]
    return ' '.join(resultwords)

def get_two_line_sentence(sentence, max_words=18):
    new_sen = sentence.split()
    return ' '.join(new_sen[ : max_words]) + '\n' + ' '.join(new_sen[ max_words : ])

def get_plot_sentence(sentence, max_words=18):
    clean_sentence = get_clean_sentence(sentence)
    if len(clean_sentence.split()) > max_words:
        return get_two_line_sentence(clean_sentence, max_words), True
    return clean_sentence, False

def show_example(dataloader, df, rows=4, cols=2):
    num_examples = cols*rows
    global_offset = 14
    font_size = 12
    max_words = 18

    transform = get_transformer('print')

    examples_df = df[ : num_examples]
    
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(10*cols, 10*rows))
    for i in range(num_examples):
        img_id = examples_df.iloc[i]['image']
        img = transform(Image.open('data/flickr8k/images/' + img_id).convert('RGB'))

        img_index = np.where(np.array(dataloader.dataset.images) == img_id)[0][0]
        captions = dataloader.dataset.__getitem__(img_index)[1]

        img = np.transpose(img, (1, 2, 0))
        sentences = get_sentences(dataloader, captions)

        ridx, cidx = i//cols, i%cols
        axs[ridx, cidx].imshow(norm(img))
        offset = global_offset
        for sen in sentences:
            sen, two_lines = get_plot_sentence(sen, max_words)
            if two_lines:
                offset += global_offset//1.5
            axs[ridx, cidx].text(2, offset, sen, fontsize=font_size, color='white', bbox=dict(facecolor='black', alpha=0.5))
            offset += global_offset            
        
        df_img = test_df[test_df['image']==img_id]
        pred = df_img['prediction'].item()
        pred, two_lines = get_plot_sentence(pred, max_words)
        if two_lines:
            offset += global_offset//1.5

        axs[ridx, cidx].text(2, offset, pred, fontsize=font_size, color='black', bbox=dict(facecolor='white', alpha=0.5))

        filter_col = [col for col in df_img if col.endswith('-gram')]
        offset = img.size(1) - ((len(filter_col) + 1) *  global_offset)

        loss = df_img['loss'].item()
        title = 'loss: {:.5f}'.format(loss)
        axs[ridx, cidx].text(2, offset, title, fontsize=font_size, color='black', bbox=dict(facecolor='white', alpha=0.5))
        offset += global_offset
        
        for col in filter_col:
            score = df_img[col].item()
            title = '{}: {:.5f}'.format(col, score)
            axs[ridx, cidx].text(2, offset, title, fontsize=font_size, color='black', bbox=dict(facecolor='white', alpha=0.5))
            offset += global_offset

        axs[ridx, cidx].axis('off')
    plt.tight_layout()
    plt.show()
```

### Good Predictions


```python
test_df = test_df.sort_values(by=['1-gram', 'loss'], ascending=False)
show_example(loader, test_df.drop_duplicates(subset=['prediction']))
```


    
![2](https://user-images.githubusercontent.com/52006798/143674627-b4f02cee-4554-47db-97c8-ee0b66937596.png)
    


### Bad Predictions


```python
test_df = test_df.sort_values(by=['1-gram', 'loss'], ascending=True)
show_example(loader, test_df.drop_duplicates(subset=['prediction']))
```


    
![3](https://user-images.githubusercontent.com/52006798/143674634-4fa06f49-911a-4c6a-b329-35d472772a87.png)
    


## Refereneces
* [A Pytorch Tutorial To Image Captioning](https://awesomeopensource.com/project/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning?categoryPage=6)
* [Image Captioning with PyTorch LSTM](https://www.kaggle.com/elanrob/image-captioning-with-pytorch-lstm)
