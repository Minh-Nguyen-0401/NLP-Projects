# custom_tokenizer.py
import spacy

nlp = spacy.load('en_core_web_sm')

def custom_tokenizer(doc):
    doc = nlp(doc)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha and not token.is_space]