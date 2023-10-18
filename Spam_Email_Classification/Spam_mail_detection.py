import matplotlib.pyplot as plt
import nltk 
import numpy as np
import pandas as pd 
import seaborn as sns 
import sklearn.naive_bayes
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.model_selection import cross_validate


data = pd.read_csv(r"D:/COLLEGEMATERIALS/Project Me/Spam_mail_detection/Spam email dataset/spam.csv",encoding="ISO-8859-1")
data['v1'] = np.where(data['v1']=='spam',1, 0)
data.head(10)

Text_list =data['v2'].tolist()
#Spam_ham_list = data['v1'].tolist()

#print(Spam_ham_list)

X_train, X_test, Y_train, Y_test = train_test_split(Text_list, data['v1'], random_state=0)

vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)

Text_list_vector = vectorizer.transform(Text_list)

X_train_vectorized = vectorizer.transform(X_train)
X_train_vectorized.toarray().shape

#print(data.head(20))

#print(X_train_vectorized)
#print(Y_train_vectorized[0])

model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_train)

predictions = model.predict(vectorizer.transform(X_test))
print("Accuracy:", 100 * sum(predictions == Y_test) / len(predictions), '%')

def cross_validation(model, _X, _y, _cv):
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }


NBresult = cross_validation(model,Text_list_vector,data['v1'],10)

print(NBresult)
print(metrics.classification_report(list(Y_test), predictions))




# Testing on New TEXT

text="congratulations, you became today's lucky winner"
k=[]
k.append(text)
k_vector = vectorizer.transform(k)

if(model.predict(k_vector)==1):
      print("Spam")
else:
      print("not spam")