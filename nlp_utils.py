import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
import spacy
import re
import spacy

nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

nlp2 = spacy.load("en_core_web_md")

def split_topics(topics):
    if pd.isna(topics): 
        return []
    return [t.strip() for t in topics.split(',')]

def open_data(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns =['index','id', 'label', 'statement', 'subject', 'speaker', 'speaker_job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context','justification']
    df.drop(columns=['index'], inplace=True)
    party_map = {
        'republican': 'right-leaning',
        'democrat': 'left-leaning',
        'libertarian': 'right-leaning',
        'tea-party-member': 'right-leaning',
        'ocean-state-tea-party-action': 'right-leaning',
        'constitution-party': 'right-leaning',
        'democratic-farmer-labor': 'left-leaning',
        'green': 'left-leaning',
        'labor-leader': 'left-leaning',
        'liberal-party-canada': 'centrist',
        'Moderate': 'centrist',
        'independent': 'centrist',
        'none': 'other',
        'organization': 'other',
        'columnist': 'other',
        'activist': 'other',
        'talk-show-host': 'other',
        'newsmaker': 'other',
        'journalist': 'other',
        'state-official': 'other',
        'business-leader': 'other',
        'education-official': 'other',
        'government-body': 'other'
    }
    df['party_category'] = df['party_affiliation'].map(party_map).fillna('other')
    df['word_count'] = df['statement'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)
    df['topic_list'] = df['subject'].apply(split_topics)
    return df

def add_sentiment_scores(df, text_col='statement'):
    """Add compound sentiment scores using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    df['statement_sentiment'] = df[text_col].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    return df

def match_counter(statement, bigram_list, threshold=70):
    """Count how many fuzzy-matched bigrams appear in a statement."""
    stat = nlp2(str(statement))
    words = [token.text.lower() for token in stat]
    bigrams = [''.join(words[i:i+2]) for i in range(len(words) - 1)]

    matches = 0
    for bigram in bigrams:
        for ref in bigram_list:
            if fuzz.ratio(bigram, ref) >= threshold:
                matches += 1
                break
    return matches

def add_bigram_features(df, text_col='statement',
                        conservative_file='top_conservative_bigrams.csv',
                        liberal_file='top_liberal_bigrams.csv'):
    """Add fuzzy bigram match counts for conservative and liberal bigrams."""
    conservative_bigrams = pd.read_csv(conservative_file)['bigram']
    liberal_bigrams = pd.read_csv(liberal_file)['bigram']

    df['conservative_bigram_count'] = df[text_col].apply(
        lambda x: match_counter(x, conservative_bigrams, threshold=70)
    )
    df['liberal_bigram_count'] = df[text_col].apply(
        lambda x: match_counter(x, liberal_bigrams, threshold=70)
    )
    return df

def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks for better embedding representation."""
    words = str(text).split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def embed_statement(text, model):
    """Compute averaged embeddings for a statement."""
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    return np.mean(embeddings, axis=0)

def add_embeddings(df, text_col='statement', model_name="all-MiniLM-L6-v2"):
    """Add averaged sentence embeddings as a new column."""
    model = SentenceTransformer(model_name)
    df['embedding'] = df[text_col].apply(lambda x: embed_statement(x, model))
    return df

def assemble_features(df, target_col='party_category'):
    """Combine sentiment, bigram counts, and embeddings into final feature matrix."""
    drop_cols = [
        'id', 'justification', 'context', 'embedding', 'statement',
        'subject', 'speaker', 'speaker_job_title', 'party_category',
        'party_affiliation'
    ]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Expand embeddings into individual numeric columns
    embeddings = np.vstack(df['embedding'].values)
    embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
    X = pd.concat([X, embedding_df], axis=1)

    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    return X, y, le

def train_and_evaluate(X, y, n_classes, cv=5):
    """Train an XGBoost model and print cross-validation accuracy."""
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=n_classes,
        max_depth=3,
        learning_rate=0.1,
        enable_categorical=True
    )

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validated accuracy ({cv}-fold): {scores.mean():.4f}")
    return model, scores

def full_pipeline(df, clean = True):
    if clean:
        df['statement'] = df['statement'].apply(clean_text)
    df = add_sentiment_scores(df)
    df = add_bigram_features(df)
    df = add_embeddings(df)
    X, y, le = assemble_features(df)
    model, scores = train_and_evaluate(X, y, len(le.classes_))
    return model, X, y, scores, le
