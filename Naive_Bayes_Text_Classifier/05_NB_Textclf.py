

import spacy 
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")



train_corpus = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'))

print(f"Train size: {len(train_corpus.data)}")
print(f"\nLabels: {train_corpus.target_names}")
print(f"\nLabel encoded: {train_corpus.target}")
print(f"\nExample Article: {train_corpus.data[0]}")

import os 
path =os.getcwd()
print(path)
# write labels to a text file to os.getcwd()
output_file = os.path.join(path, 'labels.txt')
with open(output_file, 'w') as f:
    for label in train_corpus.target_names:
        f.write(label + '\n')
print("Labels exported successfully")

            
nlp = spacy.load('en_core_web_sm')

def custom_tokenizer(doc):
    doc = nlp(doc)
    return [token.lemma_ for token in doc if token.is_stop == False and token.is_punct == False and token.is_alpha and not token.is_space]

vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)

X_train = vectorizer.fit_transform(train_corpus.data)
Y_train = train_corpus.target

import joblib

# Export the vectorizer using joblib
vectorizer = joblib.load('vectorizer.joblib')

output_file = os.path.join(path, 'vectorizer.joblib')
joblib.dump(vectorizer, output_file)
print("Vectorizer exported successfully")


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

clf = MultinomialNB()
clf.fit(X_train, Y_train)

y_train_pred = clf.predict(X_train)

train_acc = accuracy_score(Y_train, y_train_pred)

print(f"Train accuracy: {train_acc}")


from sklearn.metrics import classification_report
train_name = [Y_train]
test_rs = [y_train_pred]
target_name = [train_corpus]
report_name = ["Train"]

for idx, i in enumerate(zip(train_name, test_rs, target_name)):
    print(f"Classification report for {report_name[idx]}")
    print(classification_report(i[0], i[1], target_names=i[2].target_names))
    if idx ==0:
        print("***"*30)
    else:
        pass

# export model to a pickle file
import pickle
output_file = os.path.join(path, 'NB_textclf_model.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(clf, f)
print("Model exported successfully")





