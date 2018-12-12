import numpy as np
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
import pickle
train=pd.read_json("./data/train.json").set_index("id")
test=pd.read_json("./data/test.json").set_index("id")
train_labels=[]
for data in train["cuisine"]:
    train_labels.append(data.lower())

stemmer= SnowballStemmer("english")


def Stem(data):
    for i in range(len(data)):
        data[i]=re.sub('[0-9]+', '', data[i])
        data[i]=data[i].replace("-"," ")
        words=data[i].split()
        stemmed=[]
        for word in words:
            stemmed.append(stemmer.stem(word))
        data[i]=" ".join(x for x in stemmed)
    return data

train_input=[]
for data in train["ingredients"]:
    train_input.append(" ".join(Stem(data)).lower())
test_input=[]
for data in test["ingredients"]:
    test_input.append(" ".join(Stem(data)).lower())
    

output=open("test_feature.pkl",'wb')
pickle.dump(train_input,output)
output.close()
output=open("labels.pkl",'wb')
pickle.dump(train_labels,output)
output.close()