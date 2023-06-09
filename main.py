# # -*- coding: utf-8 -*-
# """MLA7.ipynb
#
# Automatically generated by Colaboratory.
#
# Original file is located at
#     https://colab.research.google.com/drive/1fJ3qp3tcUmG6tBJ08xh4Delno66OsKoR
# """
#
# import numpy as np
# import pandas as pd
# import seaborn as sns
#
# sns.set()
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder
#
# from google.colab import drive
#
# drive.mount('/content/drive')
#
# ds = pd.read_csv("/content/drive/MyDrive/Machine Learning Sem 6/spam.csv", encoding="ISO-8859-1")
#
# ds.info()
#
# ds.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
#
# ds['v1'] = ds['v1'].replace({'ham': 0, 'spam': 1})
# y = ds['v1']
#
# ds.head()
#
# ds.rename(columns={'v1': 'class', 'v2': 'SMS'}, inplace=True)
#
# ds.duplicated().sum()  # checking for duplicates
#
# ds = ds.drop_duplicates(keep='first')
#
# """Data Analysis"""
#
# import nltk
#
# !pip
# install
# nltk
#
# nltk.download('punkt')
#
# ds['num_char'] = ds['SMS'].apply(len)
#
# ds['num_words'] = ds['SMS'].apply(lambda x: len(nltk.word_tokenize(x)))
#
# ds['num_sent'] = ds['SMS'].apply(lambda x: len(nltk.sent_tokenize(x)))
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
# import seaborn as sns
#
# plt.figure(figsize=(10, 8))
# sns.histplot(ds[ds['class'] == 0]['num_char'])
# sns.histplot(ds[ds['class'] == 1]['num_char'])
#
# plt.figure(figsize=(10, 8))
# sns.histplot(ds[ds['class'] == 0]['num_words'])
# sns.histplot(ds[ds['class'] == 1]['num_words'])
#
# """Preprocessing"""
#
# from nltk.corpus import stopwords
#
# from nltk.stem.porter import PorterStemmer
#
#
# def pp(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     stop_words = set(stopwords.words('english'))
#     punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
#     for i in text:
#         if i not in stop_words and i not in punc:
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     ps = PorterStemmer()
#     for i in text:
#         y.append(ps.stem(i))
#
#     return " ".join(y)
#
#
# pp(ds['SMS'][0])
#
# ds['pp_text'] = ds[ds.columns[1]].apply(pp)
#
# ds.head()
#
# """Model"""
#
# from sklearn.feature_extraction.text import CountVectorizer
#
# cv = CountVectorizer(max_features=3000)
#
# X = cv.fit_transform(ds['pp_text']).toarray()
#
# y = ds['class'].values
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.3,
#                                                     random_state=1)
#
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
#
# mnb = MultinomialNB()
#
# mnb.fit(X_train, y_train)
# y_pred2 = mnb.predict(X_test)
# print(accuracy_score(y_test, y_pred2))
#
# print(y_pred2)
#
# import pickle
#
# pickle.dump(cv, open('vectorizer.pkl', 'wb'))
# pickle.dump(mnb, open('model.pkl', 'wb'))