import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

infile = open("test_feature.pkl",'rb')
X = pickle.load(infile)
infile.close()
infile = open("labels.pkl",'rb')
Y = pickle.load(infile)
infile.close()

encoder=LabelEncoder()
Y=encoder.fit_transform(Y)

features_train,features_test,labels_train,labels_test=train_test_split(X,Y, test_size=0.1)

vectorizer =TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
features_train_transformed=vectorizer.fit_transform(features_train)
features_test_transformed=vectorizer.transform(features_test)



model=LogisticRegression(dual=False,C=1,max_iter=1000,multi_class='ovr')
prediction=model.fit(features_train_transformed,labels_train).predict(features_test_transformed)
print("OneVsRestClassifier:LogisticRegression")
print(accuracy_score(labels_test,prediction))

model=LogisticRegression(dual=False,C=1,max_iter=1000,multi_class='multinomial',solver='newton-cg')
prediction=model.fit(features_train_transformed,labels_train).predict(features_test_transformed)
print("OneVsOneClassifier:LogisticRegression:newton-cg")
print(accuracy_score(labels_test,prediction))