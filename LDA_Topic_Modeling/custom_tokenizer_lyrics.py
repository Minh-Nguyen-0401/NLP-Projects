
def custom_tokenizer(doc):
    # Assume 'doc' is a spaCy Doc object
    return [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in ["ADJ", "NOUN", "VERB"]
    ]
