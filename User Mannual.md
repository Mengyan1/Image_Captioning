There are two files: train.py; generate.py

## caption_train.py
is used to train the model. 

### Input:
#### 1 words.txt 
if exits, otherwise it will be created.
It is a list of words with frequency bigger than *pra_minimun_word_freq* in the training and testing datasets. 
#### 2 train.txt
register the image name and its corresponding captionings for the training datasets. 
#### 3 train.txt
register the image name and its corresponding captionings for the test datasets.
#### 4 image_path
indicats where all the images are stored.

###Output:
#### 5 train and test accuracy
#### Best trained weights extracted by tensorflow callback

###Hypter parameters
#### 1 Sentence_Max_Length: sentences longer than this threshold will be trimmed
#### 2 IMAGE_SIZE: The input image are collected by crawler from Flickr. They are of diferent size. IMAGE_SIZE is the universal size after they are loaded by keras
#### 3 EMBEDDING_SIZE: the sizes of output from the last convolutional layer. It is how much we want to downsize the picture
#### EPOCH

## generate.py

It will generat a image caption for an individual image based on the trained model

### Inputs:

#### weights. hdf5

#### test picture

###Outputs:

#### Generated sentences



