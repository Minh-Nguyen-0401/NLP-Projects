
import pandas as pd
import numpy as np
import pickle
import joblib
# import models
from custom_tokenizer import custom_tokenizer

vectorizer = joblib.load(open('vectorizer.joblib', 'rb'))
clf = pickle.load(open('NB_textclf_model.pkl', 'rb'))
labels = pd.read_csv('labels.txt', header=None)

flag = True
while flag:
    Article = str(input("Enter an article: "))

    if Article == "exit":
        flag = False
    else:
        Article = vectorizer.transform([Article])

        idx = clf.predict(Article)[0]

        pred_article = labels.values[idx][0]
        confidence = clf.predict_proba(Article)[0][idx]

        print(f"Article belongs to {pred_article} with {confidence*100:.2f}% confidence")