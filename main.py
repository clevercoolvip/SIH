import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os


def preprocessing(df, target="Review"):
    data = df[target]
    preprocessed_data = []
    for i in range(len(data)):
        d1 = re.sub('[^a-zA-z]', ' ', data[i])
        d2 = d1.lower()
        tokens = d2.split()
        stemming = PorterStemmer()
        stemmed = [stemming.stem(word) for word in tokens if word not in set(stopwords.words('english'))]
        preprocessed_data.append(stemmed)
    dataframe = pd.DataFrame({"Preprocessed":preprocessed_data})
    dataframe.to_csv("result.csv")
    path = os.getcwd()
    print(f"Result saved to {path}")
    return preprocessed_data, dataframe


def bagOfWords(data, target="Review"):    
    data = data[target]
    v = CountVectorizer()
    df_cv = v.fit_transform(data)
    pd.DataFrame(df_cv.toarray()).to_csv("result.csv")
    path = os.getcwd()
    print(f"Result saved to {path}")
    return df_cv.toarray()
