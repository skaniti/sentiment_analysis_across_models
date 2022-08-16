import gensim
from gensim.models import word2vec
import pandas as pd
import re

import numpy as np

import warnings
warnings.filterwarnings('ignore')

#******************************************************************************************************

print("Running gensim version " + str(gensim.__version__) + ".")

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

corpus = []

for col in imdb_train.clean:
    word_list = col.split(" ")
    corpus.append(word_list)
    
full_data = pd.DataFrame(columns=['vec_1',
                                  'vec_2',
                                  'vec_3',
                                  'vec_4',
                                  'vec_5',
                                  'vec_6',
                                  'vec_7',
                                  'vec_8',
                                  'vec_9',
                                  'vec_10',
                                  'vec_11',
                                  'vec_12',
                                  'vec_13',
                                  'vec_14',
                                  'vec_15',
                                  'vec_16',
                                  'vec_17',
                                  'vec_18',
                                  'vec_19',
                                  'vec_20',
                                  'vec_21',
                                  'vec_22',
                                  'vec_23',
                                  'vec_24',
                                  'vec_25',
                                  'vec_26',
                                  'vec_27',
                                  'vec_28',
                                  'vec_29',
                                  'vec_30',
                                  'vec_31',
                                  'vec_32',
                                  'vec_33',
                                  'vec_34',
                                  'vec_35',
                                  'vec_36',
                                  'vec_37',
                                  'vec_38',
                                  'vec_39',
                                  'vec_40',
                                  'vec_41',
                                  'vec_42',
                                  'vec_43',
                                  'vec_44',
                                  'vec_45',
                                  'vec_46',
                                  'vec_47',
                                  'vec_48',
                                  'vec_49',
                                  'vec_50',
                                  'vec_51',
                                  'vec_52',
                                  'vec_53',
                                  'vec_54',
                                  'vec_55',
                                  'vec_56'
                                  ])

for i in corpus:
    model = word2vec.Word2Vec(corpus, min_count=1, vector_size=56)

    my_dict = dict({})
    for idx, key in enumerate(model.wv.index_to_key):
        my_dict[key] = model.wv.get_vector(key)
    
    vec_sum = np.sum(my_dict[key]/(len(my_dict)) for key in my_dict)
    print(vec_sum)

#vec_sum is printed row by row instead of appending to dataframe
#was having trouble with this because vec_sum is in numpy array format

'''
above code is taking vectors (one for each of 70k+ words, 
each vector 56 elems long) and summing them up for the
summed vector, take that sum and put it into a dataframe

need to divide by num of tokens in the w2v object (70k) 
(to get the average) - do this for each imdb review

vector (56 fields) label (1 field)
0.23432 0.23432 ... 0.123 "positive"
one row per review
'''