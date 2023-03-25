import pickle

import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ds = pd.DataFrame()


def pp(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in text:
        if i not in stop_words and i not in punc:
            y.append(i)

    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("SMS Spam Classifier")

# input_sms=st.text_input("Enter The message")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    ds = pd.read_csv(uploaded_file,encoding = "ISO-8859-1")
    ds.rename(columns={ds.columns[0]: 'Label', ds.columns[1]: 'SMS'}, inplace=True)  # renaming to SMS

    ds['pp_text'] = ds['SMS'].apply(pp)  # pre_proc [Working till here]

    vector_input = cv.transform(ds['pp_text']).toarray()
    pred = []
    pred = model.predict(vector_input)
    result = []
    j = 0
    for i in pred:
        if i == 1:
            result.append("Spam")
        else:
            result.append("ham")
        j += 1


    dresult = pd.DataFrame()
    dresult['Label'] = ds['Label']
    dresult['SMS']=ds['SMS']

    dresult['Classification'] = result

    st.write(dresult)
