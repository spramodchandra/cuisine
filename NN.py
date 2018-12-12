import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from scipy.sparse import csr_matrix

infile = open("test_feature.pkl",'rb')
X = pickle.load(infile)
infile.close()
infile = open("labels.pkl",'rb')
Y = pickle.load(infile)
infile.close()

encoder=LabelEncoder()
Y=encoder.fit_transform(Y)
Y=np.array(Y)
Y=Y.reshape((Y.shape[0],1))
encoder=OneHotEncoder(sparse =False)
labelsNN=encoder.fit_transform(Y)

vectorizer =TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
features_train_transformed=vectorizer.fit_transform(X)

sparse_dataset = csr_matrix(features_train_transformed)
featuresNN = sparse_dataset.todense()

#features_train, features_test, labels_train, labels_test = train_test_split(featuresNN, labelsNN, test_size=0.05)

from sklearn.neural_network import MLPClassifier
model=MLPClassifier(hidden_layer_sizes=(1000,400),activation='relu',solver='adam',max_iter=1000,early_stopping =True)
model.out_activation_='softmax'
model.fit(featuresNN,labelsNN)
print(model.n_layers_)
print(model.hidden_layer_sizes)
print(model.best_validation_score_)
