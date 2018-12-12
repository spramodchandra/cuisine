import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

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


prediction=OneVsOneClassifier(LinearSVC(dual=False,C=1,multi_class  ='crammer_singer')).fit(features_train_transformed,labels_train).predict(features_test_transformed)
print("OneVsOneClassifier:LinearSVC")
print(accuracy_score(labels_test,prediction))
prediction=OneVsRestClassifier(LinearSVC(dual=False,C=1,multi_class  ='ovr')).fit(features_train_transformed,labels_train).predict(features_test_transformed)
print("OneVsRestClassifier:LinearSVC")
print(accuracy_score(labels_test,prediction))
