import os
import subprocess
import sys
import zipfile
import warnings
import pickle

import numpy as np
import pandas as pd
import spacy
from gensim import corpora, models, similarities

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import your custom tokenizer
from custom_tokenizer_lyrics import custom_tokenizer

def download_dataset():
    """
    Downloads the Spotify Million Song Dataset from Kaggle if not already downloaded.
    """
    dataset_zip = 'spotify-million-song-dataset.zip'
    if not os.path.exists(dataset_zip):
        print("Downloading the Spotify Million Song Dataset from Kaggle...")
        try:
            # Execute the Kaggle API command to download the dataset
            subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', 'notshrirang/spotify-million-song-dataset'],
                check=True
            )
            print("Download completed successfully.")
        except subprocess.CalledProcessError as e:
            print("An error occurred while downloading the dataset.")
            print(e)
            sys.exit(1)
    else:
        print("File already downloaded.")

def unzip_dataset():
    """
    Unzips the downloaded dataset if not already unzipped.
    """
    dataset_zip = 'spotify-million-song-dataset.zip'
    extracted_file = 'spotify_millsongdata.csv'
    
    if not os.path.exists(extracted_file):
        print("Extracting the dataset...")
        try:
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                zip_ref.extractall('.')
            print(f"File extracted to current working directory: {os.getcwd()}")
        except zipfile.BadZipFile:
            print("Error: The zip file is corrupted.")
            sys.exit(1)
        except Exception as e:
            print("An unexpected error occurred while extracting the zip file.")
            print(e)
            sys.exit(1)
    else:
        print("File already extracted.")

def train_lda_model(tokenized_data, dct):
    """
    Trains an LDA model using the provided tokenized data and dictionary.
    """
    # Create Bag-of-Words representation
    bow = [dct.doc2bow(text) for text in tokenized_data]
    
    # Train LDA model
    lda_model = models.LdaModel(
        corpus=bow,
        num_topics=10,
        passes=10,
        alpha='auto',
        eta='auto',
        id2word=dct,
        random_state=0
    )
    print("LDA model trained.")
    
    # Display the topics
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")
    
    # Save the model
    lda_model.save('lda_model.model')
    print("LDA model saved as 'lda_model.model'.")
    
    return lda_model, bow

def compute_coherence(lda_model, tokenized_data, dct):
    """
    Computes the coherence score for the LDA model.
    """
    coherence_model = models.CoherenceModel(
        model=lda_model,
        texts=tokenized_data,
        dictionary=dct,
        coherence='u_mass'
    )
    coherence_score = coherence_model.get_coherence()
    print(f'Coherence Score: {coherence_score:.4f}')
    return coherence_score

def main():
    # Step 1: Download and unzip the dataset
    download_dataset()
    unzip_dataset()
    
    # Step 2: Load the dataset
    songs_df = pd.read_csv("spotify_millsongdata.csv")
    
    # Step 3: Extract and clean text data
    data = songs_df["text"].tolist()
    print("Number of songs:", len(data))
    print("\nFirst song:", data[0])
    print("\n" + "***" * 20 + "\n")
    
    # Clean text
    clean_data = [text.replace('\n', ' ').replace('\r', '') for text in data]
    print(f"Number of cleaned songs: {len(clean_data)}")
    print("\n" + "***" * 20 + "\n")
    
    # Step 4: Load spaCy model and disable unnecessary pipes
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    
    # Step 5: Tokenize the data
    print("Starting tokenization...")
    
    def batch_iterable(iterable, batch_size):
        """Yield successive batches from iterable."""
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]
    
    tokenized_data = []
    batch_size = 200  # Adjust based on system capacity
    num_processes = 2  # Sequential processing

    for idx, batch in enumerate(batch_iterable(clean_data, batch_size)):
        docs = nlp.pipe(batch, n_process=num_processes)
        for doc in docs:
            tokens = custom_tokenizer(doc)
            tokenized_data.append(tokens)
        print(f"Processed batch {idx + 1}")
    
    print(f"Number of tokenized songs: {len(tokenized_data)}")
    print("\nFirst tokenized song:", tokenized_data[0])
    print("\n" + "***" * 20 + "\n")
    
    # Step 6: Build dictionary
    dct = corpora.Dictionary(tokenized_data)
    print('Size of vocabulary before filtering:', len(dct))
    
    # Step 7: Filter tokens
    dct.filter_extremes(no_below=5, no_above=0.5)
    print('\nSize of vocabulary after filtering:', len(dct))
    
    # Example mappings
    example_mappings = [(token, dct.token2id[token]) for token in tokenized_data[0][:10] if token in dct.token2id]
    print("\nExample mappings:", example_mappings)
    
    # Save dictionary
    dct.save('dct.dict')
    print("\nDictionary saved as 'dct.dict'.")
    print("\n" + "***" * 20 + "\n")
    
    # Step 8: Train LDA model
    lda_model, bow = train_lda_model(tokenized_data, dct)
    
    # Step 9: Compute coherence score
    coherence_score = compute_coherence(lda_model, tokenized_data, dct)
    
    # Step 10: Compute cosine similarities
    lda_index = similarities.MatrixSimilarity(lda_model[bow], num_features=len(dct))
    
    # Save index
    lda_index.save('lda_index.index')
    print("\nSimilarity index saved as 'lda_index.index'.")
    print("\n" + "***" * 20 + "\n")
    
    # Save additional data for testing.py
    with open('clean_data.pkl', 'wb') as f:
        pickle.dump(clean_data, f)
    with open('bow.pkl', 'wb') as f:
        pickle.dump(bow, f)
    
if __name__ == '__main__':
    main()