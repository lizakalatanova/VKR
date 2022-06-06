# -*- coding: utf-8 -*-

!python --version

from google.colab import drive
drive.mount('/content/drive')

!pip install keras==2.4.3

!pip install tensorflow==2.4.1

import numpy as np   #Package for scientific computing and dealing with arrays
import pandas as pd  #Package providing fast, flexible and expressive data structures
import re            #re stands for RegularExpression providing full support for Perl-like Regular Expressions in Python
from bs4 import BeautifulSoup   #Package for pulling data out of HTML and XML files
from keras.preprocessing.text import Tokenizer  #For tokenizing the input sequences
from keras.preprocessing.sequence import pad_sequences  #For Padding the seqences to same length
from nltk.corpus import stopwords   #For removing filler words
from tensorflow.keras.layers import Input, LSTM, Attention, Embedding, Dense, Concatenate, TimeDistributed   #Layers required to implement the model
from tensorflow.keras.models import Model  #Helps in grouping the layers into an object with training and inference features
from tensorflow.keras.callbacks import EarlyStopping  #Allows training the model on large no. of training epochs & stop once the performance stops improving on validation dataset
import warnings  #shows warning message that may arise 

pd.set_option("display.max_colwidth", 200) #Setting the data sructure display length
warnings.filterwarnings("ignore")

reviewsData=pd.read_csv('/content/drive/My Drive/Colab Notebooks/data1.csv', delimiter=';')
print(reviewsData.shape) #Analyzing the shape of the dataset
reviewsData.head(n=10)

#Reducing the length of dataset for better training and performance
reviewsData.drop_duplicates(subset=['title'],inplace=True) #Dropping the rows with Duplicates values of 'Text'  
reviewsData.dropna(axis=0,inplace=True) #Dropping the rows with Missing values

reviewsData.info() #Getting more info on datatypes and shape of Dataset

dictionary = {"iot": "Internet of things", "ГКУ": "головный кластерный узел", "бсс": "беспроводная сенсорная сеть", 
              "sun": "smart ubiquitous networks", "sdn": "software-defined network", "ngn":"next generation network",  
              "fn":"future network", "nfv":"network function virtualization", "волс":"волокно-оптическая линия передачи", 
              "смо":"системы массового обслуживания", "son":"самоорганизующиеся сети", "цп":"циклический префикс", 
              "псп":"псевдослучайные последовательности", "абгш":"аддитивный белый Гауссовский шум", 
              "атс":"автотранспортное средство", "пкос":" программно-конфигурируемые оптические сети"}

def text_cleaner(text,num):
    newString = text.lower() 
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)  
    newString = re.sub('"','', newString)            
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^А-Яа-яa-zA-Z]", " ", newString)
    if(num==0): 
      tokens = [w for w in newString.split()] 
    else :
      tokens = newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:               
            long_words.append(i)   
    return (" ".join(long_words)).strip()
cleaned_text = []
for t in reviewsData['abstract']:
    cleaned_text.append(text_cleaner(t,0))

import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('russian')) 
def text_cleaner(text,num):
    newString = text.lower() 
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)  
    newString = re.sub('"','', newString)           
    newString = ' '.join([dictionary[t] if t in dictionary else t for t in newString.split(" ")]) 
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^А-Яа-яa-zA-Z]", " ", newString)
    if(num==0): 
      tokens = [w for w in newString.split() if not w in stop_words] 
    else :
      tokens = newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:               
            long_words.append(i)   
    return (" ".join(long_words)).strip()
cleaned_text = []
for t in reviewsData['abstract']:
    cleaned_text.append(text_cleaner(t,0))

reviewsData['title'] #Looking at the 'Text' column of the dataset

cleaned_text[:10] #Looking at the Text after removing stop words, special characters , punctuations etc.

#Summary Cleaning 
cleaned_summary = []    #Using the text_cleaner function for cleaning summary too
for t in reviewsData['title']:
    cleaned_summary.append(text_cleaner(t,1))

reviewsData['title'][:10]

cleaned_summary[:10]

!pip install rouge

from rouge import Rouge
rouge = Rouge()

reviewsData['Cleaned_abstract'] = cleaned_text  #Adding cleaned text to the dataset
reviewsData['Cleaned_title'] = cleaned_summary  #Adding cleaned summary to the dataset
#Dropping Empty Rows
reviewsData['Cleaned_title'].replace('', np.nan, inplace=True)
#Dropping rows with Missing values
reviewsData.dropna(axis=0,inplace=True)

#Before Cleaning
print("До обработки:\n")
for i in range(3):
    print("Заголовок:",reviewsData['title'][i])
    print("Аннотация:",reviewsData['abstract'][i])
    print("\n")

#Printing the Cleaned text and summary which will work as input to the model 
print("После обработки:\n")
for i in range(3):
    print("Загловок:",reviewsData['Cleaned_title'][i])
    print("Аннотация:",reviewsData['Cleaned_abstract'][i])
    print("\n")

#Data Visualization
import matplotlib.pyplot as plt
text_word_count = []
summary_word_count = []

#Populating the lists with sentence lengths
for i in reviewsData['Cleaned_title']:
      text_word_count.append(len(i.split()))
print(max(text_word_count))
for i in reviewsData['Cleaned_abstract']:
      summary_word_count.append(len(i.split()))
print(max(summary_word_count))
length_df = pd.DataFrame({'Заголовок':text_word_count, 'Аннотация':summary_word_count})
length_df.hist(bins = 30)
plt.show()

count=0
for i in reviewsData['Cleaned_title']:
    if(len(i.split())<=22):
        count=count+1
print(count/len(reviewsData['Cleaned_title']))

count=0
for i in reviewsData['Cleaned_abstract']:
    if(len(i.split())<=153):
        count=count+1
print(count/len(reviewsData['Cleaned_abstract']))

#From the above data we got an idea about maximum lengths of review and summary
max_text_len = 170
max_summary_len = 25

#Adding START and END tags to summary for better decoding
cleaned_text =np.array(reviewsData['Cleaned_abstract'])
cleaned_summary=np.array(reviewsData['Cleaned_title'])

short_text=[]
short_summary=[]

for i in range(len(cleaned_text)):
    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])
        
df=pd.DataFrame({'abstract':short_text,'title':short_summary})

df['title'] = df['title'].apply(lambda x : 'sostok '+ x + ' eostok')

#Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(np.array(df['abstract']),np.array(df['title']),test_size=0.4,random_state=0,shuffle=True)

#Preparing Tokenizer

#Text Tokenizer
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

#preparing a tokenizer for reviews on training data
X_tokenizer = Tokenizer() 
X_tokenizer.fit_on_texts(list(X_train))

#Rarewords and their coverage in review
thresh = 1  #If a word whose count is less than threshold i.e 4, then it's considered as rare word 

cnt = 0      #denotes no. of rare words whose count falls below threshold
tot_cnt = 0  #denotes size of unique words in the text
freq = 0
tot_freq = 0

for key,value in X_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1
    tot_freq=tot_freq+value
    if(value<thresh):
        cnt=cnt+1
        freq=freq+value
    
print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)

#Defining the Tokenizer with top most common words for reviews

#Preparing a Tokenizer for reviews on training data
X_tokenizer = Tokenizer(num_words=tot_cnt-cnt)   #provides top most common words
X_tokenizer.fit_on_texts(list(X_train))

#Converting text sequences into integer sequences
X_train_seq    =   X_tokenizer.texts_to_sequences(X_train) 
X_test_seq   =   X_tokenizer.texts_to_sequences(X_test)

#Padding zero upto maximum length
X_train    =   pad_sequences(X_train_seq,  maxlen = max_text_len, padding = 'post')
X_test   =   pad_sequences(X_test_seq, maxlen = max_text_len, padding = 'post')

#Size of vocabulary (+1 for padding token)
X_voc   =  X_tokenizer.num_words + 1

X_voc

#Summary Tokenizer

#Preparing a Tokenizer for summaries on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_train))

#Rarewords and their coverage in summary

thresh = 1  ##If a word whose count is less than threshold i.e 6, then it's considered as rare word 

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key,value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt+1
    tot_freq = tot_freq+value
    if(value<thresh):
        cnt = cnt+1
        freq = freq+value
    
print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)

#Defining Tokenizer with the most common words in summary

#Preparing a tokenizer for summaries on training data
y_tokenizer = Tokenizer(num_words=tot_cnt-cnt)  #provides top most common words
y_tokenizer.fit_on_texts(list(y_train))

#Converting text sequences into integer sequences
y_train_seq    =   y_tokenizer.texts_to_sequences(y_train) 
y_test_seq   =   y_tokenizer.texts_to_sequences(y_test) 

#Padding zero upto maximum length
y_train    =   pad_sequences(y_train_seq, maxlen=max_summary_len, padding='post')
y_test   =   pad_sequences(y_test_seq, maxlen=max_summary_len, padding='post')

#size of vocabulary
y_voc  =   y_tokenizer.num_words +1

y_voc

#Checking the length of training data
y_tokenizer.word_counts['sostok'],len(y_train)

#Deleting rows containing START and END tokens
#For Training set
ind=[]
for i in range(len(y_train)):
    cnt=0
    for j in y_train[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_train=np.delete(y_train,ind, axis=0)
X_train=np.delete(X_train,ind, axis=0)

#For Validation set
ind=[]
for i in range(len(y_test)):
    cnt=0
    for j in y_test[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_test=np.delete(y_test,ind, axis=0)
X_test=np.delete(X_test,ind, axis=0)

import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs, verbose=False):
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)
        def energy_step(inputs, states):
            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            e_i = K.softmax(e_i)
            if verbose:
                print('ei>', e_i.shape)
            return e_i, [e_i]
        def context_step(inputs, states):
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]
        def create_inital_state(inputs, hidden_size):
            fake_state = K.zeros_like(inputs)  
            fake_state = K.sum(fake_state, axis=[1, 2])  
            fake_state = K.expand_dims(fake_state)  
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state
        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )
        return c_outputs, e_outputs
    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

from keras import backend as K 
K.clear_session()  #Resets all state generated by Keras

latent_dim = 256
embedding_dim = 256
# Encoder
encoder_inputs = Input(shape=(max_text_len,))

#embedding layer
enc_emb =  Embedding(X_voc, embedding_dim,trainable=True)(encoder_inputs)

#encoder lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#encoder lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm 3
encoder_lstm3= LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

#Setting up the Decoder using 'encoder_states' as initial state
decoder_inputs = Input(shape=(None,))

#Embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

#Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

#Concating Attention input and Decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

#Defining the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

#Visualize the Model
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#Adding Metrics
model.compile(optimizer='rmsprop' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])

from google.colab import files
files.download('/content/drive/My Drive/Colab Notebooks/model.h5')

model.load_weights("modw.h5")

#Adding Callback
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

#Training the Model
history = model.fit([X_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size= 1)

"""# Новый раздел"""

history = model.fit([X_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=50, batch_size= 2, validation_data=([X_test,y_test[:,:-1]], y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:]))

model.save_weights("modw.h5")

files.download('modw.h5')

model.save("mods.h5")

files.download('mods.h5')

#Visualizing Accuracy 
from matplotlib import pyplot
pyplot.plot(history.history['acc'], label='train') 
pyplot.plot(history.history['val_acc'], label='test') 
pyplot.legend() 
pyplot.show()

#Visualizing Loss 
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend() 
pyplot.show()

#Building Dictionary for Source Vocabulary
reverse_target_word_index=y_tokenizer.index_word 
reverse_source_word_index=X_tokenizer.index_word 
target_word_index=y_tokenizer.word_index

#Inference/Validation Phase
#Encoding the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

#Decoder setup
#These tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

#Getting the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 

#Setting the initial states to the states from the previous time step for better prediction
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#Attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

#Adding Dense softmax layer to generate proability distribution over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat) 

#Final Decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

#Function defining the implementation of inference process
def decode_sequence(input_seq):
    #Encoding the input as state vectors
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    #Generating empty target sequence of length 1
    target_seq = np.zeros((1,1))
    #Populating the first word of target sequence with the start word
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ' '
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        #Sampling a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        #Exit condition: either hit max length or find stop word
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= 200):
            stop_condition = True

        #Updating the target sequence (of length 1)
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        #Updating internal states
        e_h, e_c = h, c

    return decoded_sentence

#Functions to convert an integer sequence to a word sequence for summary as well as reviews 
def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

def text_cleaner(text,num):
    newString = text.lower() 
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)  
    newString = re.sub('"','', newString)            
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^А-Яа-яa-zA-Z]", " ", newString)
    if(num==0): 
      tokens = [w for w in newString.split()] 
    else :
      tokens = newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:               
            long_words.append(i)   
    return (" ".join(long_words)).strip()
cleaned_text = []
for t in reviewsData['abstract']:
    cleaned_text.append(text_cleaner(t,0))

#Summary Cleaning 
cleaned_summary = []    #Using the text_cleaner function for cleaning summary too
for t in reviewsData['title']:
    cleaned_summary.append(text_cleaner(t,1))

cleaned_text

cleaned_summary

reviewsData['title'][:10]

print(reverse_source_word_index[1021])

!pip install anvil-uplink

import anvil.server

anvil.server.connect("RIIBA374EKCC42BXPOMHMP54-CEZRKHQPC4VGSQUE")


@anvil.server.callable
def predict_title(text_area_3):
  X_seq =  X_tokenizer.texts_to_sequences([text_area_3]) 
  #Padding zero upto maximum length
  X_seq2    =   pad_sequences(X_seq,  maxlen = max_text_len)
  print(X_seq2)
  st = decode_sequence(X_seq2.reshape(1,max_text_len))
  st = re.sub("  ","",st)
  return st

anvil.server.wait_forever()
