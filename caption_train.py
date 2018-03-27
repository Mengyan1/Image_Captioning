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


IMAGE_PATH = 'data/flickr30k_images'
TRAIN_CAPTION_PATH = 'data/train.txt'
TEST_CAPTION_PATH = 'data/test.txt'
WORDS_PATH = 'data/words.txt' 


SENTENCE_MAX_LENGTH = 100
EMBEDDING_SIZE = 256
IMAGE_SIZE = 224
EPOCH = 4
CHECK_ROOT = 'checkpoint/'

if not os.path.exists(CHECK_ROOT):
    os.makedirs(CHECK_ROOT)

class Data_generator(object):
    '''
    This is a self-defined data generator that can be read by model.fit_generator
    1 It load the corresponding image to each caption
    2 It convets words to index and then index to one-hot encoding vectors
    3 It post-padding the captions in the index stage
    4 It shuffle the dataset before each epoch
    '''
    def __init__(self, pra_batch_size=20, pra_minimun_word_freq=2):
        self.minimum_word_freq = pra_minimun_word_freq # remove unique words

        #two lists of image name and two lists of its corresponding captioning
        self.train_image_paths, self.train_captions, \
         self.test_image_paths, self.test_captions = self.get_name_caption()
        self.train_captions_index = self.caption2index(self.train_captions)
        self.test_captions_index = self.caption2index(self.test_captions)

        self.batch_size = pra_batch_size
        self.train_steps_epoch = len(self.train_image_paths)//pra_batch_size # steps per epoch,round down
        self.test_steps_epoch = len(self.test_image_paths)//pra_batch_size # steps per epoch
     
    def get_dictionary(self, pra_captions):
        ''' 
        Generate a dictionary mapping all words and indexs.
        Return:
            words2index: word->index dictionary 
            index2words: index->word dictionary
        '''
        if not os.path.exists(WORDS_PATH):
            words = set(' '.join(pra_captions).split(' '))
            with open(WORDS_PATH, 'w') as writer:
                writer.write('\n'.join(words))
        else:
            with open(WORDS_PATH, 'r') as reader:
                words = [line.strip() for line in reader.readlines()]

        self.voc_size = len(words) #vocabulary size, size of all acceptable words

        words2index = dict((word, index) for index, word in enumerate(words, start=0))
        index2words = dict((index, word) for index, word in enumerate(words, start=0))
        return words2index, index2words

    def caption2index(self, pra_captions):
        '''
        Generate a list to store the index for each words in a list of captions
        according to the word2index dictionary
        '''
        words2index, _ = self.get_dictionary(pra_captions)
        captions = [x.split(' ') for x in pra_captions]
        index_captions = [[words2index[w] for w in cap if w in words2index.keys()] for cap in captions]
        return index_captions

    def index2caption(self, pra_index):
        '''

        '''
        _, index2words = self.get_dictionary('')
        captions = [' '.join([index2words[w] for w in cap]) for cap in pra_index]
        return captions 

    def convert2onehot(self, pra_caption):
        '''
        Onehot encoding the caption indexes for one image
        '''
        captions = np.zeros((len(pra_caption), self.voc_size))
        for index, cap in enumerate(pra_caption, start=0):
            captions[index, cap] = 1
        return np.array(captions)

    def get_epoch_steps(self):      
        return self.train_steps_epoch, self.test_steps_epoch

    def generate(self, pra_train=True):
        '''
        Continuously generate training or testing data.
        1, Post-padding captionings
        2, Shuffle the dataset befor each epoch
        3, 
            pra_train = True : generate training data
            pra_train = False : generate testing data
        '''
        while True:
            if pra_train:
                # Shuffle training data at the beginning of each epoch.
                shuffle_index = np.random.permutation(len(self.train_image_paths))
                image_name_list = np.array(self.train_image_paths)[shuffle_index]
                image_caption_list = np.array(self.train_captions)[shuffle_index]
                image_caption_index_list = np.array(self.train_captions_index)[shuffle_index]
                print ('The number of training images is:', len(image_name_list))
           
            else:
                image_name_list = self.test_image_paths
                image_caption_list = self.test_captions
                image_caption_index_list = self.test_captions_index

            image_caption_index_list = Sequence.pad_sequences(image_caption_index_list, maxlen=SENTENCE_MAX_LENGTH, padding='post')
            
            input_image_list = []
            input_caption_list = []
            target_caption_list = []

            for index, (image_name, image_caption) in enumerate(zip(image_name_list, 
                image_caption_index_list), start=1):
                input_image = Image.img_to_array(Image.load_img(image_name, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3)))
                input_caption_onehot = self.convert2onehot(image_caption)
                target_caption_onehot = np.zeros_like(input_caption_onehot)
                
                # Creating targets for LSTM by moving the input one-step forward. 
                # 0~n-1 rows of target_caption is the same as 1~n rows of the input_caption
                target_caption_onehot[:-1] = input_caption_onehot[1:]
                
                input_image_list.append(input_image)
                input_caption_list.append(input_caption_onehot)
                target_caption_list.append(target_caption_onehot)

                if len(input_image_list) == self.batch_size:
                    tmp_images = np.array(input_image_list)
                    tmp_captions = np.array(input_caption_list)
                    tmp_targets = np.array(target_caption_list)
                    input_image_list = []
                    input_caption_list = []
                    target_caption_list = []
                    yield [preprocess_input(tmp_images), tmp_captions], tmp_targets
    
    def get_name_caption(self):
        '''
        1，Load training and testing data from files, return two captioning list
           and the corresponding image_name_list.
        2，Add a <START> and <END> to the beginning and the end of each sentence respectively.
        3，Creat a word file containing high frequency (frequency > pra_minimun_word_freq) words 
        
        Returns:   
            train_caption_list: corresponding training captions
            test_caption_list: corresponding testing captions
            train_image_name_lists: all paths of training images
            test_image_name_lists: all paths of testing images       
        '''
        with open(TRAIN_CAPTION_PATH, 'r') as reader:
            content = [line.strip().split('\t') for line in reader.readlines()] # each row is one line in train.txt
            train_image_name_list = [os.path.join(IMAGE_PATH, line[0].split('#')[0]) for line in content]
            train_caption_list = ['<START> {} <END>'.format(line[1].lower()) for line in content]

        with open(TEST_CAPTION_PATH, 'r') as reader:
            content = [x.strip().split('\t') for x in reader.readlines()]
            test_image_name_list = [os.path.join(IMAGE_PATH, line[0].split('#')[0]) for line in content]
            test_caption_list = ['<START> {} <END>'.format(line[1].lower()) for line in content]    

        #Count the frequency of each word, and get rid of the unique words
        all_words = ' '.join(train_caption_list+test_caption_list).split(' ')
        words_num = Counter(all_words)
        words = [x for x in words_num if words_num[x]>=self.minimum_word_freq]
        print('{} unique words (all).'.format(len(words_num)))
        print('{} unique words (count>={}).'.format(len(words), self.minimum_word_freq))
        with open(WORDS_PATH, 'w') as writer:
            writer.write('\n'.join(words))

        return train_image_name_list, train_caption_list, \
                test_image_name_list, test_caption_list


class Image_Caption(object):
    '''
    '''
    def __init__(self, pra_voc_size):
        '''
        Model achitecture is defined.
        1 Load VGG16 with top_layer
        2 Add a fully connected convolution layer to downsize the image feature
        3 Concatenate image features with word vectors
        4 Add a one-to-one LSTM layer
        5 Use categorical_crossentropy as loss function and rmsprop as optimizer 
        '''  
        self.voc_size = pra_voc_size

        # Load VGG16 wegiths pre-trained on ImageNet and keep fixed. 
        base_model = VGG16(weights='imagenet', include_top=True)
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        for layer in base_model.layers[1:]:
            layer.trainable = False

        # Add a fully connected layer to reduce the image feature size
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
        
        self.model = model
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def train_val_model(self, pra_datagen):
        # callback: draw curve on TensorBoard
        tensorboard = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=True)

        #callback: save the weight with the highest validation accuracy
        filepath=os.path.join(CHECK_ROOT, 'weights-improvement-{val_acc:.4f}-{epoch:04d}.hdf5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
        # train model 
        self.model.fit_generator(
            pra_datagen.generate(pra_train=True), 
            steps_per_epoch=pra_datagen.get_epoch_steps()[0], 
            epochs=2, 
            validation_data=pra_datagen.generate(pra_train=False), 
            validation_steps=pra_datagen.get_epoch_steps()[1],
            callbacks=[tensorboard,checkpoint])



if __name__ == '__main__':
    my_generator = Data_generator()
    model = Image_Caption(my_generator.voc_size)
    model.train_val_model(my_generator)
