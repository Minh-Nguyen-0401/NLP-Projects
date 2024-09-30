from custom_tokenizer_lyrics import custom_tokenizer
import spacy
from gensim import corpora, models, similarities
import os
import pickle
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


dct = corpora.Dictionary.load('dct.dict')
lda_index = similarities.MatrixSimilarity.load('lda_index.index')
lda_model = models.LdaModel.load('lda_model.model')

song_info = pd.read_csv('spotify_millsongdata.csv')
song_info = song_info.drop(columns=['link'], axis=1)


# Load additional trained data
with open('clean_data.pkl', 'rb') as f:
    clean_data = pickle.load(f)
with open('bow.pkl', 'rb') as f:
    bow = pickle.load(f)

def get_similar_songs(song_bow, top_n=5, first_m_words=300):
    """
    Returns the top N most similar songs to the input song.
    """
    similar_songs = lda_index[lda_model[song_bow]]
    top_n_docs = sorted(enumerate(similar_songs), key=lambda item: -item[1])[1:top_n+1]
    
    # Returns a list of tuples: (song_id, similarity_score, song_excerpt)
    return [
        (entry[0], entry[1], clean_data[entry[0]][:first_m_words])
        for entry in top_n_docs
    ]

def new_song_similarity():
    new_song = input("Enter lyrics of a new song: ")
    # Tokenize the new song
    doc = nlp(new_song)
    new_tokens = custom_tokenizer(doc)
    new_song_bow = dct.doc2bow(new_tokens)
    similar_songs = get_similar_songs(new_song_bow)
    
    # Display the results
    for idx, (song_id, sim_score, song_excerpt) in enumerate(similar_songs):
        print(f"\nRank {idx + 1}:")
        print(f"Similarity Score: {sim_score:.4f}")

        #disply also artist name, song name by mapping text column with song_id
        print(f"Artist: {song_info.loc[song_id, 'artist']}")
        print(f"Song Name: {song_info.loc[song_id, 'song']}")
        print(f"Song Lyrics: {song_excerpt}")
    
    return similar_songs

if __name__ == '__main__':
    flag = True
    while flag:
        new_song_similarity()
        user_input = input("Do you want to enter another new song? (y/n): ").strip().lower()
        
        if user_input == 'y':
            flag = True
        else:
            flag = False
