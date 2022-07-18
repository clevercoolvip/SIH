import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim


'''
extarcting features and labels from a dataframe, if given by user.
Other wise, using autoai by vipin, to extract features and labels
'''
def extract_X_Y(df, x=None, y=None):
    if x is not None:
        X = df[x]
        return X
    if y is not None:
        Y = df[y]
        return Y 


'''
Preprocessing the extracted text field:
It includes:
    Regular Expression for removing numbers and special characters,
    Changing the words to lower case,
    tokenisation of every word of every sentences,
    Stemming the words and removing the Past, continuous, etc verb forms
    Output:
        1. List of preprocessed text corpus
        2. Panda Dataframe of preprocessed corpus
'''
def preprocessTextField(df):
    resultant_corpus=[]
    for i in range(len(df)):
        review_df = re.sub('[^a-zA-z]', ' ', df[i])
        l = review_df.lower()
        tokenize = l.split()
        ps = PorterStemmer()
        stemmed = [ps.stem(word) for word in tokenize if word not in set(stopwords.words('english'))]
        resultant_corpus.append(stemmed)
    corpus_df = pd.DataFrame({"Preprocessed":resultant_corpus})
    return resultant_corpus, corpus_df


'''
Converting Word to vector using cosine similarities
And saving the model.
'''
def word2vector(corpus, minimum_word=2, cpu_threads=4, save_at="word2vector_model.model"):
    model = gensim.models.Word2Vec(
        window=10,
        min_count=minimum_word,
        workers=cpu_threads
    )
    model.build_vocab(corpus, progress_per=1000)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print(model.corpus_count)
    print(model.epochs)
    model.save(save_at)
    return model
