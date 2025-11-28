from bs4 import BeautifulSoup
from newspaper import Article
import glob
import pandas as pd
import requests
import spacy

key = "ea04f85bdd4b485c9c93f5faca413d05"
url = "https://newsapi.org/v2/everything"  
topic = "Columbus Day"

def ingest_articles(topic):
    cond = {"q": topic,"language": "en", "pageSize": 75, "apiKey": key, "sortBy": "relevancy"}

    call = requests.get(url, params=cond)
    raw_art = call.json()
    article_list = []
    for a in raw_art.get("articles", []):
        raw_url = a.get("url")
        source_name = a.get("source", {}).get("name", "Unknown")
        
        raw_text = None
        try:
            curr_art = Article(raw_url)
            curr_art.download()
            curr_art.parse()
            raw_text = curr_art.text
        except Exception as e:
            raw_text = None
        
        article_list.append({
            "statement": raw_text,
            "source": source_name
        })

    df = pd.DataFrame(article_list)
    df = df[df['statement'].notna() & (df['statement'].str.strip() != '')]
    df['statement'] = (
        df['statement']
        .astype(str)  
        .str.replace(r'\s+', ' ', regex=True) 
        .str.strip() 
    )
    return df

def ingest_articles_LP(topic):
    cond = {
        "q": topic,
        "language": "en",
        "pageSize": 75,
        "apiKey": key,
        "sortBy": "relevancy"
    }

    call = requests.get(url, params=cond)
    raw_art = call.json()

    articles = []
    nlp = spacy.load("en_core_web_sm")

    for a in raw_art.get("articles", []):
        raw_url = a.get("url")
        source_name = a.get("source", {}).get("name", "Unknown")
        author = a.get("author") or source_name
        title = a.get("title") or ""
        desc = a.get("description") or ""

        # download + parse article body
        try:
            art_obj = Article(raw_url)
            art_obj.download()
            art_obj.parse()
            full_text = art_obj.text
        except:
            full_text = None

        if not full_text:
            continue

        # Clean statement
        statement = " ".join(full_text.split())

        # Extract possible location
        doc = nlp(statement[:1000])  # limit for speed
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        state = gpes[0] if gpes else ""

        # Construct LIAR-PLUS-like row
        row = {
            "id": 0,
            "label": "",  # unknown truth value
            "statement": statement,
            "subjects": [topic],
            "speaker": author,
            "speaker_job": "Journalist" if author != source_name else "News Source",
            "state": state,
            "party": "none",
            "barely_true_counts": 0,
            "false_counts": 0,
            "half_true_counts": 0,
            "mostly_true_counts": 0,
            "pants_on_fire_counts": 0,
            "context": title + " | " + desc,
            "justification": raw_url,
            "source": source_name,
            "published_at": a.get("publishedAt")
        }

        articles.append(row)

    return pd.DataFrame(articles)
