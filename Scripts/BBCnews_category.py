# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:14:16 2022

@author: pc
"""

import os 
import re 
import pickle
import json
import datetime
import pandas as pd 
import numpy as np 

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping 

from Modules_BBCnews_category import ModelCreation,Model_Evaluation

#%% Statics

CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')

#%% Data Loading
df = pd.read_csv(CSV_URL)

#%% Data Inspection

# To check the forst 10 rows of dataset
df.head(10)
# Category and test column is identified

df.info()
# 0:category 1: text
# Dtype: object

df['category'].unique()
# There are 5 categories ; tech,business,sport,entertainment,politics

# To check duplicated values inside 
df.duplicated().sum() # 99 duplicated values
df[df.duplicated()]

# To check null values
df.isnull().any()
# No null values


# To visualise the 'category'(target variable) by using countplot
sns.countplot(df.category)
# From visualisation, the distribution of count by each category is not
# perfecly balanced but can say it is nearly balanced as only sport and business
# has the highest count else just have 400

#%% Data Cleaning 

# To drop duplicate values 
df = df.drop_duplicates()

#remove HTML tags
text = df['text'].values # Feature: x
category = df['category'].values # Target: y
# Put .values to make sure the shape of dataset is linked are the same

for index,tex in enumerate(text):
    # remove html tags 
    # ? dont be greedy
    # * zero or more occurences 
    # any character except new line (/n)
    
    text[index] = re.sub('<.*?>',' ',tex)
    
    # convert into lower case
    # remove numbers 
    # ^ means NOT 
    text[index] = re.sub('[^a-zA-Z]',' ', tex).lower().split()

#%% Features Selection

# This section nothing to select since this dataset has no features and 
# it is refering to text classification and sequences so we proceed to preprocessing 

#%% Preprocessing (TOKENIZER)

# To convert all the vocab into number within 10000 words
vocab_size = 10000
oov_token = 'OOV'
max_len = 180

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text)# to learn all the words
word_index = tokenizer.word_index
# print(word_index) to show the words_num dictionary

train_sequences = tokenizer.texts_to_sequences(text) # to convert into numbers

#%% (PADDING) for x variable: text

length_of_text = [len(i) for i in train_sequences] # list comprehension
print(np.median(length_of_text)) # to get the number of max length for padding


padded_text = pad_sequences(train_sequences,maxlen=max_len,padding='post',
              truncating='post') # refer to x variable


#%% (ONEHOTENCODER) for y variable: category

ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

#%% Train Test Split

x_train,x_test,y_train,y_test = train_test_split(padded_text,category,
                                                 test_size=0.3,
                                                 random_state=123)


x_train = np.expand_dims(x_train,axis=-1) # to reshape since the x train is in 2D
x_test = np.expand_dims(x_test,axis=-1)

#%% Model Development (LSTM)

# import modules for ez alter, just need to change value and layer 
mc = ModelCreation()
model = mc.embed_lstm_layer(x_train,num_node=128)

# embedding_dim = 64 
# output_node = len(np.unique(y_train,axis=0))

# model = Sequential()
# model.add(Input(shape=(np.shape(x_train)[1]))) # Input shape=(180,1)np.shape(x_train)[1:]
# model.add(Embedding(vocab_size, embedding_dim)) # embedding size doesnt accept array (180,1)
# model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
# # model.add(LSTM(128,return_sequences=(True)))
# model.add(Dropout(0.3))
# model.add(LSTM(128))
# model.add(Dropout(0.3))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(output_node,'softmax'))
# model.summary()

# model.compile(loss='categorical_crossentropy',optimizer='adam',
#               metrics='acc')

# To plot flowchart of model layer
plot_model(model,show_layer_names=(True),show_shapes=(True))

# Stopping Callbacks 
# Tensorboard
early_stopping_callback = EarlyStopping(monitor='loss', patience=3)
LOG_PATH = os.path.join(os.getcwd(),'Logs')
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
log_dir = datetime.datetime.now()

hist = model.fit(x_train,y_train,batch_size=64,epochs=50,
                 validation_data=(x_test,y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])


# Sequential_1 getting acc only 0.35% and it is very low with 1 dense layer 
# Added 1 more dense layer and training it if let say can increase the accuracy 
# But, it is getting lower so i will go try use embedding and bidrectional lstm model 
# in order to increase the accuracy of the training. p/s: Sequential_3 i got error so proceed with Sequential_4
# Sequential_4 with 10 epochs, the accuracy increase to 0.56% 
# For Sequential_5, let say try 50 epochs, and also use callback functions
# After training, the accuracy increases tremendously and val_acc also increase
# val_loss getting decreases as the epoch stops at 24, the val_acc is 76%


#%% Plot Visualisation

# To potray training and validation for each loss and accuracy
hist.history.keys()
# Use modules for ez alter
hist_me = Model_Evaluation()
hist_me.plot_hist_graph(hist)


# From the plotting visualisation, on the epoch loss section, 
# Starting at epoch 2, the validation become overfitting even though im already use early stopback
# maybe the training epochs is low, well we can try to improvise it by increase the epochs
# also we can try to overcome this accuracy by increasing the dropout rate
# Other DL architecture also can be use such as transformer,BERT Model,GPT3 Model

#%% Model evaluation 

y_true = y_test 
y_pred = model.predict(x_test)

y_true = np.argmax(y_true,axis=1) # to convert 0/1 
y_pred = np.argmax(y_pred,axis=1)

print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))

# Conclusion
# The f1 score,recall,precision accuracy can be obtain from classification report
# by import confusion matrix,accuracy_score module to summarise the training statistic
# the model accuracy is 76% which is still acceptable to do model deployment
# Nevertheless, we can still imporove the training and fit it with other model architecture
# Proceed to Model H5 save

#%% Model saving

#H5 model save
model.save(MODEL_SAVE_PATH)

#Initialise token_json
token_json = tokenizer.to_json()

#to save tokenizer as dictionary
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

# To save ohe
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)








