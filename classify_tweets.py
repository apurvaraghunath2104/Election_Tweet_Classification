import pandas as pd
import nltk
import re
from sklearn import *
import time
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

# nltk.download('punkt')

def clean_data(unclean_tweets):
    cleaned_data = []
    for tweet in unclean_tweets:
        tweet = tweet.lower()
        tweet = re.sub(r'@\w+', r'', tweet)
        tweet = re.sub('<[^<]+?>', '', tweet)
        tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
        tweet = re.sub(r'(\s)@\w+', r'', tweet)
        tweet = re.sub(r'[<>!#@$:.,%\?-]+', r'', tweet)
        tweet = tweet.replace("'", "").replace("\"","")
        words = nltk.word_tokenize(tweet.lower())
        cleaned_data.append(words)
    return cleaned_data


#*********************TRAIN*********************************

#input
file='training-Obama-Romney-tweets.xlsx'

df_obama = pd.read_excel(file,'Obama')
df_romney = pd.read_excel(file,'Romney')

df_obama_copy = df_obama[['Anootated tweet','Unnamed: 4']]
df_obama_copy = df_obama_copy.drop(0)
df_obama_copy = df_obama_copy.rename(columns = {"Anootated tweet": "tweet", "Unnamed: 4": 'class' })
df_obama_copy = df_obama_copy.dropna()
df_obama_copy = df_obama_copy[(df_obama_copy['class'].isin((1,-1,0)))]


obama_index_list = df_obama_copy.index.tolist()

obama_unclean_tweets = df_obama_copy['tweet']
obama_class = df_obama_copy['class']

obama_unclean_tweets = obama_unclean_tweets.tolist()
obama_class_train = obama_class.tolist()

obama_tweets = clean_data(obama_unclean_tweets)

df_romney_copy = df_romney[['Anootated tweet','Unnamed: 4']]
df_romney_copy = df_romney_copy.drop(0)
df_romney_copy = df_romney_copy.rename(columns = {"Anootated tweet": "tweet", "Unnamed: 4": 'class' })
df_romney_copy = df_romney_copy.dropna()
df_romney_copy = df_romney_copy[(df_romney_copy['class'].isin((1,-1,0)))]


romney_index_list = df_romney_copy.index.tolist()

romney_unclean_tweets = df_romney_copy['tweet']
romney_class = df_romney_copy['class']

romney_unclean_tweets = romney_unclean_tweets.tolist()
romney_class_train = romney_class.tolist()

romney_tweets = clean_data(romney_unclean_tweets)


file_test='final-testData-no-label-Obama-tweets(1).xlsx'

df_obama_test = pd.read_excel(file_test,header=None)


# df_obama_copy_test = df_obama_test.rename(columns = {"Anootated tweet": "tweet"})
df_obama_copy_test = df_obama_test.dropna()
# print("df_obama_copy_test****",df_obama_copy_test)

df_obama_copy_test.columns =['index','tweet']
# print("df_obama_copy_test after naming****",df_obama_copy_test)
obama_index_list_test = df_obama_copy_test.index.tolist()

# print("df_obama_copy_test",df_obama_copy_test.iloc[:,1])
# df_obama_copy_test.reset_index(drop=True)
# df_obama_copy_test['tweet'] = df_obama_copy_test
obama_unclean_tweets_test = df_obama_copy_test['tweet']
# obama_unclean_tweets_test = df_obama_copy_test
# print("obama_unclean_tweets_test before",obama_unclean_tweets_test)

obama_unclean_tweets_test = obama_unclean_tweets_test.values.tolist()
# print("obama_unclean_tweets_test after",obama_unclean_tweets_test)

obama_tweets_test = clean_data(obama_unclean_tweets_test)

file_test='final-testData-no-label-Romney-tweets(1).xlsx'

df_romney_test = pd.read_excel(file_test,header=None)


# df_romney_copy_test = df_romney_test.rename(columns = {"Anootated tweet": "tweet"})
df_romney_copy_test = df_romney_test.dropna()
df_romney_copy_test.columns =['index','tweet']
# print("df_romney_copy_test after naming****",df_romney_copy_test)
romney_index_list_test = df_romney_copy_test.index.tolist()


romney_unclean_tweets_test = df_romney_copy_test['tweet']
# romney_unclean_tweets_test = df_romney_copy_test


romney_unclean_tweets_test = romney_unclean_tweets_test.values.tolist()
# print("romney_unclean_tweets_test after",romney_unclean_tweets_test)
romney_tweets_test = clean_data(romney_unclean_tweets_test)


df_obama_copy['tweet'] = obama_tweets
y = df_obama_copy['class']
df_obama_copy_train = df_obama_copy['tweet']
y_train_obama = y
y_train_obama=y_train_obama.astype('int')

# df_obama_copy_test['tweet'] = obama_tweets_test
# df_obama_copy_test = df_obama_copy_test['tweet']

df_obama_copy_test = obama_tweets_test
# df_obama_copy_test = df_obama_copy_test


df_romney_copy['tweet'] = romney_tweets
y = df_romney_copy['class']
df_romney_copy_train = df_romney_copy['tweet']
y_train_romney = y
y_train_romney=y_train_romney.astype('int')

# df_romney_copy_test['tweet'] = romney_tweets_test
# df_romney_copy_test = df_romney_copy_test['tweet']
df_romney_copy_test = romney_tweets_test
# df_romney_copy_test = df_romney_copy_test['tweet']

print("Length of cleansed Obama test",len(obama_tweets_test))
print("Length of cleansed Romney test",len(romney_tweets_test))

'''
#************inital train data split into train and validation************88
df_obama_copy['tweet'] = obama_tweets
y = df_obama_copy['class']
df_obama_copy_train = df_obama_copy['tweet'][:4923]
df_obama_copy_val = df_obama_copy['tweet'][4923:]
y_train = y[:4923]
y_train=y_train.astype('int')
y_val = y[4923:]
y_val=y_val.astype('int')

df_romney_copy['tweet'] = romney_tweets
y_romney = df_romney_copy['class']
df_romney_copy_train = df_romney_copy['tweet'][:5083]
df_romney_copy_val = df_romney_copy['tweet'][5083:]
y_train_romney = y_romney[:5083]
y_train_romney=y_train_romney.astype('int')
y_val_romney = y_romney[5083:]
y_val_romney=y_val_romney.astype('int')

'''


def create_features(train):
    vectorizer = TfidfVectorizer(min_df=2, tokenizer=token_function, lowercase=False)
    train_vectors = vectorizer.fit_transform(train)
    return vectorizer,train_vectors


def token_function(docs):
    return docs

'''
#initial vectorization for train and validation

tfidf, X_train= create_features(df_obama_copy_train)
X_val = tfidf.transform(df_obama_copy_val)


tfidf_romney, X_train_romney= create_features(df_romney_copy_train)
X_val_romney = tfidf_romney.transform(df_romney_copy_val)
'''



tfidf, X_train_obama= create_features(df_obama_copy_train)
X_test_obama = tfidf.transform(df_obama_copy_test)

tfidf, X_train_romney= create_features(df_romney_copy_train)
X_test_romney = tfidf.transform(df_romney_copy_test)


'''
#training and Predicting the inital train and validation

clf_svm = svm.SVC(kernel='linear', C=0.91)
clf_svm.fit(X_train, y_train)
obama_pred_svm = clf_svm.predict(X_val)
obama_pred_train_svm = clf_svm.predict(X_train)
accuracy_svm = accuracy_score(y_val,obama_pred_svm)
labels = [1,-1]
precision_svm = metrics.precision_score(y_val,obama_pred_svm,average=None,labels=labels)
recall_svm = metrics.recall_score(y_val,obama_pred_svm,average=None,labels=labels)
f1_score_svm = metrics.f1_score(y_val,obama_pred_svm,average=None,labels=labels)
print("Obama SVM: \nOverall Acurracy: ",accuracy_svm,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision_svm[i]))
    print("Recall of %s class: %f" %(lbl[i],recall_svm[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1_score_svm[i]),"\n")

clf_svm_romney = svm.SVC(kernel='linear', C=0.91)
clf_svm_romney.fit(X_train_romney, y_train_romney)
romney_pred_svm = clf_svm_romney.predict(X_val_romney)
romney_pred_train_svm = clf_svm_romney.predict(X_train_romney)
accuracy_svm_romney = accuracy_score(y_val_romney,romney_pred_svm)
labels = [1,-1]
precision_svm_romney = metrics.precision_score(y_val_romney,romney_pred_svm,average=None,labels=labels)
recall_svm_romney = metrics.recall_score(y_val_romney,romney_pred_svm,average=None,labels=labels)
f1_score_svm_romney = metrics.f1_score(y_val_romney,romney_pred_svm,average=None,labels=labels)
print("Romney SVM: \nOverall Acurracy: ",accuracy_svm_romney,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision_svm_romney[i]))
    print("Recall of %s class: %f" %(lbl[i],recall_svm_romney[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1_score_svm_romney[i]),"\n")

'''



print("RESULTS FOR TEST OBAMA*************************************")
clf_svm_obama = svm.SVC(kernel='linear', C=0.91)
clf_svm_obama.fit(X_train_obama, y_train_obama)
obama_pred_svm = clf_svm_obama.predict(X_test_obama)
print("Prediction length for Obama",len(obama_pred_svm))


print("RESULTS FOR TEST ROMNEY*************************************")
clf_svm_romney = svm.SVC(kernel='linear', C=0.91)
clf_svm_romney.fit(X_train_romney, y_train_romney)
romney_pred_svm = clf_svm_romney.predict(X_test_romney)
print("Prediction length for Romney",len(romney_pred_svm))


#Writing results into file
obama_pred_list = obama_pred_svm.tolist()
romney_pred_list = romney_pred_svm.tolist()

result_path_obama = "obama.txt"
result_path_romney = "romney.txt"
output_file_obama = open(result_path_obama,"w")
output_file_obama.write("51\n")
for i in range(0,len(obama_index_list_test)):
    output_file_obama.write(str(obama_index_list_test[i]+1)+";;"+str(obama_pred_list[i])+"\n")
output_file_romney = open(result_path_romney,"w")
output_file_romney.write("51\n")
for i in range(0,len(romney_index_list_test)):
    output_file_romney.write(str(romney_index_list_test[i]+1)+";;"+str(romney_pred_list[i])+"\n")


'''
Testing on different models and find the best one
#********************Navie bayes OBAMA****************************
clf_nb = naive_bayes.BernoulliNB()
clf_nb.fit(X_train, y_train)
obama_pred = clf_nb.predict(X_val)
accuracy_nb = metrics.accuracy_score(y_val,obama_pred)
labels = [1,-1]
precision_nb = metrics.precision_score(y_val,obama_pred,average=None,labels=labels)
recall_nb = metrics.recall_score(y_val,obama_pred,average=None,labels=labels)
f1_score_nb = metrics.f1_score(y_val,obama_pred,average=None,labels=labels)
print("Obama Navie Bayes: \nOverall Acurracy: ",accuracy_nb,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision_nb[i]))
    print("Recall of %s class: %f" %(lbl[i],recall_nb[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1_score_nb[i]),"\n")

#********************Navie bayes ROMNEY****************************
clf_nb_romney = naive_bayes.BernoulliNB()
clf_nb_romney.fit(X_train_romney, y_train_romney)
romney_pred = clf_nb_romney.predict(X_val_romney)
accuracy_nb_romney = metrics.accuracy_score(y_val_romney,romney_pred)
labels = [1,-1]
precision_nb_romney = metrics.precision_score(y_val_romney,romney_pred,average=None,labels=labels)
recall_nb_romney = metrics.recall_score(y_val_romney,romney_pred,average=None,labels=labels)
f1_score_nb_romney = metrics.f1_score(y_val_romney,romney_pred,average=None,labels=labels)
print("Romney Navie Bayes: \nOverall Acurracy: ",accuracy_nb_romney,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision_nb_romney[i]))
    print("Recall of %s class: %f" %(lbl[i],recall_nb_romney[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1_score_nb_romney[i]),"\n")



#******************DecisionTreeClassifier OBAMA******************
clf_dt = tree.DecisionTreeClassifier(max_depth = 40)
clf_dt.fit(X_train, y_train)
obama_pred = clf_dt.predict(X_val)
accuracy_dt = metrics.accuracy_score(y_val,obama_pred)
labels = [1,-1]
precision_dt = metrics.precision_score(y_val,obama_pred,average=None,labels=labels)
recall_dt = metrics.recall_score(y_val,obama_pred,average=None,labels=labels)
f1_score_dt = metrics.f1_score(y_val,obama_pred,average=None,labels=labels)
print("Obama Decision Tree: \nOverall Acurracy: ",accuracy_dt,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision_dt[i]))
    print("Recall of %s class: %f" %(lbl[i],recall_dt[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1_score_dt[i]),"\n")

#******************DecisionTreeClassifier ROMNEY******************
clf_dt_romney = tree.DecisionTreeClassifier(max_depth = 40)
clf_dt_romney.fit(X_train_romney, y_train_romney)
romney_pred = clf_dt_romney.predict(X_val_romney)
accuracy_dt_romney = metrics.accuracy_score(y_val_romney,romney_pred)
labels = [1,-1]
precision_dt_romney = metrics.precision_score(y_val_romney,romney_pred,average=None,labels=labels)
recall_dt_romney = metrics.recall_score(y_val_romney,romney_pred,average=None,labels=labels)
f1_score_dt_romney = metrics.f1_score(y_val_romney,romney_pred,average=None,labels=labels)
print("Romney Decision Tree: \nOverall Acurracy: ",accuracy_dt_romney,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision_dt_romney[i]))
    print("Recall of %s class: %f" %(lbl[i],recall_dt_romney[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1_score_dt_romney[i]),"\n")


#*********************************Random forest OBAMA**********************************************
clf_rf = ensemble.RandomForestClassifier(criterion='entropy', n_jobs = 6)
clf_rf.fit(X_train, y_train)
obama_preds = clf_rf.predict(X_val)
accuracy_rf = metrics.accuracy_score(y_val,obama_preds)
labels = [1,-1]
precision_rf = metrics.precision_score(y_val,obama_preds,average=None,labels=labels)
recall_rf = metrics.recall_score(y_val,obama_preds,average=None,labels=labels)
f1_score_rf = metrics.f1_score(y_val,obama_preds,average=None,labels=labels)
print("Obama Random Forest: \nOverall Acurracy: ",accuracy_rf,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision_rf[i]))
    print("Recall of %s class: %f" %(lbl[i],recall_rf[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1_score_rf[i]),"\n")

#*********************************Random forest ROMNEY**********************************************
clf_rf_romney = ensemble.RandomForestClassifier(criterion='entropy', n_jobs = 6)
clf_rf_romney.fit(X_train_romney, y_train_romney)
romney_preds = clf_rf_romney.predict(X_val_romney)
accuracy_rf_romney = metrics.accuracy_score(y_val_romney,romney_preds)
labels = [1,-1]
precision_rf_romney = metrics.precision_score(y_val_romney,romney_preds,average=None,labels=labels)
recall_rf_romney = metrics.recall_score(y_val_romney,romney_preds,average=None,labels=labels)
f1_score_rf_romney = metrics.f1_score(y_val_romney,romney_preds,average=None,labels=labels)
print("Romney Random Forest: \nOverall Acurracy: ",accuracy_rf_romney,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision_rf_romney[i]))
    print("Recall of %s class: %f" %(lbl[i],recall_rf_romney[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1_score_rf_romney[i]),"\n")
'''
