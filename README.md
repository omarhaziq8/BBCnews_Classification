# BBCnews_Classification

<a><img alt='python' src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a><img alt = 'image' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
<a><img alt='tf' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>
<a><img alt='keras' src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"></a>
<a><img alt='numpy' src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"></a>
<a><img alt='pandas' src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"></a>
<a><img alt='sk-learn' src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"></a>


<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>


**Description** : Text sequences datasets/BBC news categorization, to categorize unseen articles into 5 categories 

**Algorithm Model** : Deep Learning method->> LSTM, Long Short Term Memory, BIdirectional 

**Preprocessing step** : TOKENIZER, PADDING, ONE HOT ENCODER

**Objectives** : 
1) To categorize dataset into 5 categories: Sport,Business,Tech,Entertainment,Politics
2) To achieve at least 70% accuracy of model training development

**Flowchart Model** :

<img src="Snipping_Training/model.png" alt="Girl in a jacket" style="width:500px;height:600px;"> 

### Exploratory Data Analysis (EDA)
1) Data Loading
2) Data Inspection
3) Data Cleaning
4) Features Selection
5) Pre-Processing

**Model evaluation** :

`Classification_report`
`accuracy_score`
`Confusion_Matrix`
`Model_train_test_split`
`json`
`pickle`
`EDA`

**Discussion** :

 🟠The dataset is started by loading dataset from raw websites, EDA techniques to inspect duplicated,Null values insdie dataset
 
 🟠Cleaning the dataset by droping duplicated values, use tokenizer to convert all the vocab into number within 10000 words
 
 🟠Padding is introduced to text to ensure all the sequnces have the same lenght
 
 🟠Onehotencoder funtions for y variable(category) in order for Deep Learning training
 
 🟠Sequential_1 getting acc only 0.35% and it is very low with 1 dense layer
 
 🟠Added 1 more dense layer and training it if let say can increase the accuracy 
 
 🟠But, it is getting lower so i will go try use embedding and bidrectional lstm model 
 
 🟠In order to increase the accuracy of the training. p/s: Sequential_3 i got error and i fixed then proceed with Sequential_4
 
 🟠Sequential_4 with 10 epochs, the accuracy increase to 0.56% while Sequential_5, let say try 50 epochs, and also use callback functions
 
 🟠After training, the accuracy increases tremendously and val_acc also increase val_loss getting decreases as the epoch stops at 24, the val_acc is 76%
 
 🟠# From the plotting visualisation, on the epoch loss section, starting at epoch 2, the validation become overfitting even though im already use early stopback, maybe the training epoch is low, well we can try to improvise it by increase the number of epoch
 
 🟠# Besides, we can try to overcome this accuracy by increasing the dropout rate.Other DL architecture also can be use such as transformer,BERT Model,GPT3 Model
 
 **Conclusion** :
 
🗡️ The f1 score,recall,precision accuracy can be obtain from classification report

🗡️ by import confusion matrix,accuracy_score module to summarise the training statistic

🗡️ the model accuracy is 76% which is still acceptable to do model deployment

🗡️ Nevertheless, we can still imporove the training and fit it with other model architecture

🗡️ Proceed to Model H5 save
 
**Dataset** :

![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

[Datasets](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv)

**Credits** : 

[Credit to susanli2016](https://github.com/susanli2016)


<h3 align="left">Connect with me:</h3>
<p align="left">www.linkedin.com/in/omarhaziq
</p>


**Enjoy Coding!** 🚀
 
