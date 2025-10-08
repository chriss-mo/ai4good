import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
import spacy

nlp = spacy.load("en_core_web_md")

def add_sentiment_scores(df, text_col='statement'):
    """Add compound sentiment scores using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    df['statement_sentiment'] = df[text_col].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    return df

def match_counter(statement, bigram_list, threshold=70):
    """Count how many fuzzy-matched bigrams appear in a statement."""
    stat = nlp(str(statement))
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

def full_pipeline(df):
    df = add_sentiment_scores(df)
    df = add_bigram_features(df)
    df = add_embeddings(df)
    X, y, le = assemble_features(df)
    model, scores = train_and_evaluate(X, y, len(le.classes_))
    return model, X, y, scores, le
