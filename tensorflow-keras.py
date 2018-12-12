import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from tensorflow import keras
from scipy.sparse import csr_matrix


infile = open("test_feature.pkl",'rb')
X = pickle.load(infile)
infile.close()
infile = open("labels.pkl",'rb')
Y = pickle.load(infile)
infile.close()

encoder=LabelEncoder()
Y=encoder.fit_transform(Y)
labelsNN = keras.utils.to_categorical(Y)


vectorizer =TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
features_train_transformed=vectorizer.fit_transform(X)

sparse_dataset = csr_matrix(features_train_transformed)
featuresNN = sparse_dataset.todense()
X_trainNN, X_testNN, y_trainNN, y_testNN = train_test_split(featuresNN, labelsNN, test_size=0.1)


model = keras.models.Sequential()
model.add(keras.layers.Dense(1000,input_dim = X_trainNN.shape[1] ,activation = 'relu'))
model.add(keras.layers.Dense(600,activation = 'relu'))
model.add(keras.layers.Dense(200,activation = 'relu'))
model.add(keras.layers.Dense(20,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.fit(X_trainNN,y_trainNN,epochs=2)
print("Accuracy" ,model.evaluate(X_testNN,y_testNN)[1])
