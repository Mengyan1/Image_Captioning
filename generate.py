import os
import numpy as np
from collections import Counter

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Merge, Flatten, RepeatVector, TimeDistributed, Concatenate
from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing import image as Image
from keras.preprocessing import sequence as Sequence
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model, to_categorical

from keras.layers.wrappers import Bidirectional


WORDS_PATH = 'model1/words.txt'
WEIGHTS_PATH = 'model1/weights.hdf5'

SENTENCE_MAX_LENGTH = 100 
EMBEDDING_SIZE = 256
IMAGE_SIZE = 224

class Image_Caption(object):
    def __init__(self, pra_voc_size=12503):
        self.voc_size = pra_voc_size
        self.model = self.create_model()
        self.model.load_weights(WEIGHTS_PATH)

    def generate_caption(self, image_path):
        start_word = ['<START>']
        start_index = self.caption2index(start_word)
        end_word = ['<END>']
        end_index = self.caption2index(end_word)
        print("The index of <START> is: %s" %start_index)
        print("The index of <END> is: %s" %end_index)
        end_vector = self.convert2onehot(end_index)
       
        image = Image.img_to_array(Image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        image = np.expand_dims(image, 0)
        image = preprocess_input(image)

        index_list = start_index[0]
        for i in range(SENTENCE_MAX_LENGTH):
            
            print("Index_list", index_list)
            padded_index = Sequence.pad_sequences([index_list], maxlen=SENTENCE_MAX_LENGTH, padding='post')
            current_onehot = self.convert2onehot(np.transpose(padded_index))
            current_onehot = np.array([current_onehot])
            next_onehot = self.model.predict([image, current_onehot],batch_size=1)
            print("The shape of next_onehot is :",np.shape(next_onehot))
            next_index = np.argmax(next_onehot[0,i])
            index_list.append(next_index)
            print("In %sth step, the predict index is %s" %(i, next_index))
            print("Output sentence:", self.index2caption([index_list]))
            if (next_index == end_index):
                break
            print()

    def create_model(self):

        base_model = VGG16(weights='imagenet', include_top=True)
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        for layer in base_model.layers[1:]:
            layer.trainable = False

        image_model = Sequential()
        image_model.add(base_model)
        image_model.add(Dense(EMBEDDING_SIZE, activation='relu'))
        image_model.add(RepeatVector(SENTENCE_MAX_LENGTH))

        language_model = Sequential()
        language_model.add(LSTM(128, input_shape=(SENTENCE_MAX_LENGTH, self.voc_size), return_sequences=True))
        language_model.add(TimeDistributed(Dense(128)))

        model = Sequential()
        model.add(Merge([image_model, language_model], mode='concat'))
        model.add(LSTM(1000, return_sequences=True))
        model.add(TimeDistributed(Dense(self.voc_size, activation='softmax')))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_dictionary(self, pra_captions):
        if not os.path.exists(WORDS_PATH):
            words = set(' '.join(pra_captions).split(' '))
            with open(WORDS_PATH, 'w') as writer:
                writer.write('\n'.join(words))
        else:
            with open(WORDS_PATH, 'r') as reader:
                words = [x.strip() for x in reader.readlines()]

        words2index = dict((w, ind) for ind, w in enumerate(words, start=0))
        index2words = dict((ind, w) for ind, w in enumerate(words, start=0))
        return words2index, index2words

    def caption2index(self, pra_captions):
        words2index, index2words = self.get_dictionary(pra_captions)
        captions = [x.split(' ') for x in pra_captions]
        index_captions = [[words2index[w] for w in cap if w in words2index.keys()] for cap in captions]
        return index_captions

    def index2caption(self, pra_index):
        words2index, index2words = self.get_dictionary('')
        captions = [' '.join([index2words[w] for w in cap]) for cap in pra_index]
        return captions

    def convert2onehot(self, pra_caption):
        captions = np.zeros((len(pra_caption), self.voc_size))
        for ind, cap in enumerate(pra_caption, start=0):
            captions[ind, cap] = 1
        return np.array(captions)

    def decode(self, onehot):
        index = np.argmax(onehot, axis=1)
        print(index)
        captions = self.index2caption([index])
        return captions

if __name__ == '__main__':
    model = Image_Caption()
    #image = "data/flickr30k_images/667626.jpg"
    test_image = "data/flickr30k_images/86341446.jpg"
    model.generate_caption(test_image)
