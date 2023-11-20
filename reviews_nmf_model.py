import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import re
import string
import emoji
import langid
import chardet
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
import spacy
from joblib import load


st.set_page_config(
    layout="wide",
    page_title="Restaurant Review Summarizer",
    page_icon=":fork_and_knife:",
    menu_items={
        "Get help": "mailto:salimkilinc@yahoo.com",
        "About": "For More Information\n" + "https://github.com/salimkilinc"
    }
)

background_image = """
<style>
    .stApp {
        background-image: url('https://lh3.googleusercontent.com/pw/ADCreHfgyKIstYsBgwW8QI57-RMCeDAut_ZoIPt4epEpBGygU5D1U2QEsEBJ5xVywCZ4PdGvLtDMFHGxcZt-nitd0FBfDNuhtK87kBfdUBX_x9j1aNSBFlnV-wjG0L002wT7Lf4pPL2cUKWG4jPEd3b-JY24=w1922-h1442-s-no-gm?authuser=0');
        background-size: cover;
        background-repeat: no-repeat;
    }
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

nmf_model = load('reviews_nmf_model.pkl')
vectorizer = load('vectorizer.pkl')

try:
    words.ensure_loaded()
except LookupError:
    nltk.download('words')
    
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words_set = load('stop_words_set.pkl')
nlp = spacy.load('en_core_web_sm')

combinations = [f"{letter1}{letter2}" for letter1 in 'abcdefghijklmnopqrstuvwxyz' for letter2 in 'abcdefghijklmnopqrstuvwxyz']
combinations_set = set(combinations)

def detect_encoding(text):
    result = chardet.detect(text.encode())
    return result['encoding'], result['confidence']

def lemmatize_word(word):
    return lemmatizer.lemmatize(word, pos='n')

def lemmatize_and_filter(word):
    lemma = lemmatizer.lemmatize(word, pos='n')
    return lemma if lemma in set(nltk.corpus.words.words()) else ''

def remove_repeating_letters(s):
    words = s.split()
    result = []
    for word in words:
        new_word = re.sub(r'(.)\1{2,}', r'\1', word)
        result.append(new_word)
    return ' '.join(result)

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words_set])

def remove_unimportant_words(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def remove_binaries(text):
    return ' '.join([word for word in text.split() if word.lower() not in combinations_set])

def lemmatize_and_remove_adverbs(sentence):
    doc = nlp(sentence)
    lemmatized_tokens = [token.lemma_ if token.pos_ != 'ADV' else '' for token in doc]
    lemmatized_tokens = [token for token in lemmatized_tokens if token]
    return ' '.join(lemmatized_tokens)

def preprocess_function(text):
    combinations_set = set(combinations)
    
    text = text.replace("<br />", "")
    text = text.replace("\n\n", "")
    text = text.replace("…", " ")
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text.lower())
    text = re.sub(' +', ' ', text)
    text = remove_repeating_letters(text)
    text = emoji.replace_emoji(text, replace='')
    text = remove_repeating_letters(text)
    
    text = ' '.join([word for word in text.split() if langid.classify(word)[0] == 'en'])
    
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    text = re.sub('[^a-zA-Z\s]', '', text)
    text = ' '.join([lemmatize_word(word) for word in text.split()])
    text = ' '.join([lemmatize_and_filter(word) for word in text.split()])
    text = remove_stopwords(text)
    text = remove_unimportant_words(text)
    text = remove_binaries(text)
    text = lemmatize_and_remove_adverbs(text)
    
    custom_stopwords = ['excellent', 'outstanding', 'superb', 'fantastic', 'terrific', 'marvelous', 'wonderful',
                        'exceptional', 'admirable', 'splendid', 'poor', 'inferior', 'subpar', 'mediocre', 'lousy',
                        'terrible', 'awful', 'horrible', 'abysmal', 'dismal', 'overwhelming', 'exemplary',
                        'extraordinary', 'remarkable', 'unparalleled', 'exceptional', 'unsurpassed', 'superlative',
                        'peerless', 'incomparable', 'atrocious', 'dreadful', 'deplorable', 'appalling', 'catastrophic',
                        'abominable', 'monstrous', 'detestable', 'reprehensible', 'unbearable', 'good', 'bad', 'great',
                        'eatery', 'diner', 'bistro', 'cafe', 'brasserie', 'tavern', 'cafeteria', 'grill', 'pub',
                        'trattoria', 'location', 'spot', 'venue', 'area', 'site', 'locale', 'setting', 'region',
                        'space', 'position', 'restaurant', 'place', 'sommeli', 'cha', 'second', 'minute', 'hour', 'day', 'week',
                        'fortnight', 'month', 'year', 'decade', 'century', 'millennium', 'moment', 'quarter', 'half',
                        'nighttime', 'midnight', 'noon', 'future', 'era', 'epoch', 'age', 'period', 'interval', 'schedule',
                        'calendar', 'clock', 'watch', 'stopwatch', 'timer', 'chronometer', 'timepiece', 'o\'clock', 'a.m.',
                        'p.m.', 'yesterday', 'today', 'tomorrow', 'dawn', 'dusk', 'zone', 'daylight', 'lunar', 'solar',
                        'calendar', 'leap', 'gregorian', 'julian', 'sidereal', 'equinox', 'solstice', 'century', 'millennium', 'san']
    text = ' '.join([word for word in text.split() if word not in custom_stopwords])
    
    return text


def classify_new_document(new_document):
    preprocessed_document = preprocess_function(new_document)

    new_doc_word = vectorizer.transform([preprocessed_document])

    new_doc_topic = nmf_model.transform(new_doc_word)

    topic_labels = {
        0: 'Efficient Dining Service',
        1: 'Quality Food and Atmosphere',
        2: 'Wine and Culinary Delights',
        3: 'Ordering and Waiting',
        4: 'Attentive Dining Experience',
        5: 'Delicious Dinner Atmosphere',
        6: 'Timely Dining Experience',
        7: 'Flavorful Menu Options',
        8: 'Overall Dining Experience',
        9: 'Hotel Breakfast Delights'
    }

    topic_distribution = pd.DataFrame(new_doc_topic.round(10), columns=topic_labels.values())
    dominant_topic = topic_distribution.idxmax(axis=1)
    topic_distribution.insert(0, "Dominant Topic", dominant_topic)
    
    return topic_distribution

def main():
    st.title('Restaurant Review Summarizer')

    new_document = st.text_area('Describe your restaurant visit:', '', height=200)

    if st.button('Summarize'):
        with st.spinner('Summarizing...'):
            result = classify_new_document(new_document)
            st.write(result)

            dominant_topic = result["Dominant Topic"].values[0]
            probability_column = result[dominant_topic].values[0] * 100

            st.markdown(f"Your input aligns with **{dominant_topic}**, taking the culinary stage at **{probability_column:.2f}%**.")

if __name__ == '__main__':
    main()
