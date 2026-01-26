# AI4Good: Misinformation Detection System

This project detects and ranks misinformation in text using a hybrid predictive-generative approach. The architecture combines feature engineering with a BERT-based neural classifier.

## Architecture Overview

### Core Pipeline: Statistical + Neural Approach

The system analyzes text through **four factuality factors**:

1. **BERT/Transformer Embeddings** (`utils/mxnet_utils.py`, `BERTClassifier`)
   - Fine-tuned BERT-base-uncased with metadata fusion
   - Takes text + contextual features (author, topic, history)
   - Outputs 6-class veracity predictions: false, barely-true, half-true, mostly-true, pants-fire, true

2. **Political Bias Detection** (`utils/nlp_utils.py`, `add_bigram_features()`)
   - Fuzzy-matched bigram counting (conservative vs liberal phrase lists)
   - Sources: Stanford congressional speech data (`data/top_conservative_bigrams.csv`, `data/top_liberal_bigrams.csv`)
   - Returns: stat count (NER entities), conservative/liberal bigram matches

3. **Sentiment/Sensationalism** (`notebooks/webpage_runner.ipynb`)
   - VADER sentiment analyzer for emotional intensity
   - Spam detection via fine-tuned BERT-tiny (mrm8488/bert-tiny-finetuned-sms-spam-detection)

4. **Generative AI Analysis** (`gemini_prompt_refined.ipynb`, `webpage_runner.ipynb`)
   - Uses Gemini/OpenAI APIs with specialized prompts
   - Takes feature vectors from (1-3) as context for LLM analysis

## Key Files & Data Flow

**Training Data**: `data/liar-plus/` (train2.tsv, val2.tsv, test2.tsv) + manually labeled articles
- Original format: 15 TSV columns including speaker metadata, veracity counts, context
- Processed: `data/lp-clean/` contains cleaned statements with labels

**Model Architecture** (`utils/mxnet_utils.py`):
```
Text → [BERT encoder] → [pooler_output: 768-dim CLS token]
                          ↓
Metadata (topics, author, job, location, affiliation) → [embedding + MLP] → [author features]
History counts (veracity distribution) → [MLP] → [history features]
                          ↓
[Concatenate: CLS + author features + history features] → [Classifier MLP] → logits (6 classes)
```

**Feature Engineering** (`utils/nlp_utils.py`):
- Text cleaning: lowercasing, URL removal, lemmatization (spaCy en_core_web_md)
- Sentiment: VADER polarity scores
- Political bigrams: fuzzy string matching (threshold=70)
- Embeddings: averaged sentence-transformers over text chunks (chunk_size=200, overlap=50)

**Inference Pipeline** (`notebooks/webpage_runner.ipynb`):
1. Load pretrained model from `checkpoints/best.pth`
2. Extract veracity probabilities + bias/sentiment scores
3. Pass feature vector to Gemini/OpenAI with specialized prompt
4. Expose via Gradio web interface

## Developer Workflows

### Setup
```bash
pip install -r req.txt
# Download pretrained model and place in checkpoints/best.pth (not in repo due to size)
```

### Key Notebooks

- **`full_pred_model.ipynb`**: End-to-end training pipeline
  - Loads LIAR-PLUS data via `nlp_utils.open_data()`
  - Transforms to feature vectors using `feature_extraction_transform()`
  - Trains BERTClassifier with DataLoader + collate_fn for batching
  
- **`webpage_runner.ipynb`**: Production inference + Gradio web app
  - Calls `run_truth_model()` with article text
  - Computes bias/sentiment scores via `analyze_for_genai()`
  - Prompts LLM with feature context
  - Serves web UI via Gradio

- **`rag_poc.ipynb`**: Retrieval-Augmented Generation from PDFs
  - Demonstrates ChromaDB for document chunking/retrieval

### Model Loading Patterns

All notebooks follow this pattern for loaded pretrained weights:
```python
import pickle
with open("../data/vocabs.pkl", "rb") as f:
    vocabs = pickle.load(f)
# Vocabularies: topic_vocab, author_vocab, job_vocab, location_vocab, affiliation_vocab, label_map

net = BERTClassifier(...)
net.load_state_dict(torch.load('../checkpoints/best.pth', map_location=device))
```

**Note**: Checkpoint path is relative from notebook location (`../checkpoints/`).

## Critical Patterns

**Metadata Vocabularies**: All categorical features (author, job, location, affiliation) are looked up in hardcoded vocabularies. If introducing new data, must generate/update `vocabs.pkl` or use placeholder IDs (vocab[unknown_key] returns 0).

**Data Format Conversion**: LIAR-PLUS reader expects 15-column TSV. Mapping in `nlp_utils.open_data()` hardcodes column order—changing order breaks parsing.

**Text Cleaning Philosophy**: Heavy preprocessing (stopword removal, lemmatization) used in some pipelines but not in BERT inference (BERT is pretrained on raw text). Don't over-clean text before feeding to BERT model.

**Fuzzy Bigram Matching**: Threshold=70 chosen empirically for political bigrams. Lowering increases false positives; raising reduces sensitivity.

## External Dependencies

- **Transformers/PyTorch**: BERT fine-tuning, spam detection
- **spaCy**: NLP preprocessing (en_core_web_md for lemmatization + NER)
- **News API** (`ingestor_utils.py`): Article scraping (requires API key)
- **Gemini/OpenAI APIs**: Generative analysis (requires API keys in notebook cells)
- **Gradio**: Web UI for inference

## Common Issues

1. **"Loaded vocabs and label_map" not printing**: vocabs.pkl missing or corrupted—regenerate from full pipeline
2. **BERT tokenizer mismatch**: Always use `BertTokenizer.from_pretrained('bert-base-uncased')` to match model
3. **Device mismatches**: Explicitly move tensors to device (cpu/cuda) before forward pass; see `run_truth_model()`
4. **API key cells**: `webpage_runner.ipynb` has empty `api_key = ""` cell—fill with actual key before running Gemini/OpenAI calls
