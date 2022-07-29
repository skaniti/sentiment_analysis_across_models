from gensim.models import word2vec, FastText
import pandas as pd
import re

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import plotly.graph_objects as go

import numpy as np

import warnings
warnings.filterwarnings('ignore')

#******************************************************************************************************

imdb_train = pd.read_csv('csv/imdb_train.csv')
imdb_test = pd.read_csv('csv/imdb_test.csv')


clean_txt = []

for w in range(len(imdb_train.text)):
    desc = imdb_train['text'][w].lower()

    #remove punctuation
    desc = re.sub('[^a-zA-Z]', ' ', desc)

    #remove tags
    desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)

    #remove digits and special chars
    desc=re.sub("(\\d|\\W)+"," ",desc)
    clean_txt.append(desc)

imdb_train['clean'] = clean_txt
print(imdb_train.head())

corpus = []

for col in imdb_train.clean:
    word_list = col.split(" ")
    corpus.append(word_list)
    
print(corpus[0:1])

model = word2vec.Word2Vec(corpus, min_count=1, vector_size=56)

print(model)

vec_sum = np.sum([model[key] for key in model.index_to_key], axis=0)
print(vec_sum)
#need to divide by num of tokens in the w2v object (70k) (to get the average) - do this for each imdb review

'''
above code is taking vectors (one for each of 70k+ words, each vector 56 elems long) and summing them up for the summed vector
take that sum and put it into a dataframe

vector (56 fields) label (1 field)
0.23432 0.23432 ... 0.123 "positive"
one row per review

'''

'''

#pass the embeddings to PCA
X = model.wv
pca = PCA(n_components=2)
result = pca.fit_transform(X)

#create df from the pca results
pca_df = pd.DataFrame(result, columns = ['x','y'])

#add the words for the hover effect
words = list(model.vocab)

pca_df['word'] = words
pca_df.head()

N = 1000000

fig = go.Figure(data=go.Scattergl(
   x = pca_df['x'],
   y = pca_df['y'],
   mode='markers',
   marker=dict(
       color=np.random.randn(N),
       colorscale='Viridis',
       line_width=1
   ),
   text=pca_df['word'],
   textposition="bottom center"
))

fig.show()

'''