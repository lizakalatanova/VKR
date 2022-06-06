# -*- coding: utf-8 -*-

!pip install simpletransformers

from transformers import MBartTokenizer, MBartForConditionalGeneration
model_name = "IlyaGusev/mbart_ru_sum_gazeta"

tokenizer = MBartTokenizer.from_pretrained(model_name)

model = MBartForConditionalGeneration.from_pretrained(model_name)

model.to("cuda")

!pip install rouge
from rouge import Rouge
rouge = Rouge()

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
reviewsData=pd.read_csv('/content/drive/My Drive/Colab Notebooks/data.csv',encoding="cp1251", delimiter=';')
print(reviewsData.shape) #Analyzing the shape of the dataset
reviewsData.head(n=10)

import numpy as np


!pip install anvil-uplink

import anvil.server

anvil.server.connect("RIIBA374EKCC42BXPOMHMP54-CEZRKHQPC4VGSQUE")

!pip install lexrank
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
!pip install razdel
from re import S
from razdel import tokenize, sentenize
import math

@anvil.server.callable
def predict_iris(text_area_1):
  article_text=text_area_1;
  lxr = LexRank(article_text, stopwords=STOPWORDS['ru'])
  tokenise = list(sentenize(article_text))
  summary1 = lxr.get_summary([i.text for i in tokenise], summary_size=math.ceil(len(tokenise)/2), threshold=None)
  full_data = ' '.join(summary1)
  print(full_data)
  input_ids = tokenizer(
    [article_text],
    max_length=1000,
    truncation=True,
    return_tensors="pt",
  )["input_ids"].to("cuda")
  output_ids = model.generate(
     input_ids=input_ids,
      min_length=80
  )[0]
  summary = tokenizer.decode(output_ids, skip_special_tokens=True)
  print(summary)
  return summary

from gensim.summarization import keywords

import re

!pip install pymorphy2

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = list(stopwords.words('russian'))

stop_words.append('это')

stop_words.append('который')

@anvil.server.callable
def predict_sum(text_area_1, lengt):
  article_text=text_area_1;
  lxr = LexRank(article_text, stopwords=STOPWORDS['ru'])
  tokenise = list(sentenize(article_text))
  if lengt < 1:
    sumsize=math.ceil(len(tokenise)*lengt)
  else:
    sumsize=lengt
  if (len(tokenise)> sumsize): 
 	  summary1 = lxr.get_summary([i.text for i in tokenise], summary_size=sumsize, threshold=None)
  	full_data = ' '.join(summary1)
  else:
	   full_data = "Выберите меньшее количество предложений. Количество предложений в исходном тексте составляет - " + str(len(tokenise)))

  return full_data

@anvil.server.callable
def predict_key(text_area_1, coun):
    article_text = re.sub("[^A-Za-zА-Яа-я]", " ", text_area_1)
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    def lemmatize(text):
        words = article_text.split() # разбиваем текст на слова
        res = list()
        for word in words:
           p = morph.parse(word)[0]
           res.append(p.normal_form)
        return res
    clean_word_data = [w for w in lemmatize(article_text) if w.lower() not in stop_words]
    clean_word = " ".join(clean_word_data)
    if (len(clean_word)> coun): 
    	full_data = keywords(clean_word, words=coun)
	    full_data = re.sub("\n",", ", full_data)
    else:	
   	  full_data = "Выберите меньшее количество слов. Количество слов в исходном тексте составляет - " + str(len(clean_word))
    return full_data

anvil.server.wait_forever()
