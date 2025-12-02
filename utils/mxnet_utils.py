import pandas as pd
import numpy as np 
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
padding_id = tokenizer.pad_token_id

def load_tsv_data(filepath):
    # The original code used field_indices=[1, 2, ..., 13]
    # This implies skipping the first column (index 0) and reading the next 13.
    try:
        data = pd.read_csv(filepath,
                           sep='\t',
                           header=None,
                        #    usecols=range(1, 14),
                           on_bad_lines='skip',
                           encoding='utf-8')
        data.dropna(subset=[2], inplace=True)
        # Convert to list of lists to match gluonnlp.data.TSVDataset output
        return data.values.tolist()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

class CustomVocab:
    def __init__(self, counter, min_freq=1, unknown_token='<unk>'):
        self.unknown_token = unknown_token
        self._token_to_idx = {unknown_token: 0}
        self._idx_to_token = [unknown_token]

        idx = 1
        for token, freq in counter.items():
            if freq >= min_freq:
                if token not in self._token_to_idx:
                    self._token_to_idx[token] = idx
                    self._idx_to_token.append(token)
                    idx += 1
        self._unknown_idx = self._token_to_idx[unknown_token]

    def __getitem__(self, token):
        return self._token_to_idx.get(token, self._unknown_idx)

    def __len__(self):
        return len(self._idx_to_token)

    def __repr__(self):
        return f"CustomVocab(size={len(self)})"
    
class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BERTClassifier(nn.Module):
    def __init__(self, bert, num_topics, num_authors, num_jobs, num_locations, num_affiliations, num_classes,
                 embed_dim,
                 author_dropout, author_mlp_layers, author_mlp_hidden,
                 history_dropout, history_mlp_layers, history_mlp_hidden):
        super(BERTClassifier, self).__init__()
        self.bert = bert

        # Note: BERT hidden size is 768
        bert_hidden_size = bert.config.hidden_size

        # self.topic_embed = nn.Linear(num_topics, embed_dim) # Use one-hot encoding for topics
        self.topic_embed = nn.Sequential(
            nn.Linear(num_topics, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1)
        )

        self.author_embed = nn.Embedding(num_embeddings=num_authors, embedding_dim=embed_dim)
        self.job_embed = nn.Embedding(num_embeddings=num_jobs, embedding_dim=embed_dim)
        self.location_embed = nn.Embedding(num_embeddings=num_locations, embedding_dim=embed_dim)
        self.affiliation_embed = nn.Embedding(num_embeddings=num_affiliations, embedding_dim=embed_dim)

        author_feature_map_layers = []
        author_feature_map_layers.append(nn.Dropout(author_dropout))
        author_input_dim = embed_dim * 5 # topic + author + job + location + affiliation
        for _ in range(author_mlp_layers):
            author_feature_map_layers.append(nn.Linear(author_input_dim, author_mlp_hidden))
            author_feature_map_layers.append(nn.LeakyReLU(0.1))
            author_feature_map_layers.append(nn.Dropout(author_dropout))
            author_input_dim = author_mlp_hidden # Input for next layer
        self.author_feature_map = nn.Sequential(*author_feature_map_layers)

        history_feature_map_layers = []
        history_input_dim = 6 # 5 proportions + 1 uncertainty
        for _ in range(history_mlp_layers):
            history_feature_map_layers.append(nn.Linear(history_input_dim, history_mlp_hidden))
            history_feature_map_layers.append(nn.LeakyReLU(0.1))
            history_feature_map_layers.append(nn.Dropout(history_dropout))
            history_input_dim = history_mlp_hidden # Input for next layer
        self.history_feature_map = nn.Sequential(*history_feature_map_layers)

        # extra layer used for classification
        classifier_input_dim = bert_hidden_size + author_input_dim + history_input_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, inputs, segment_types, attention_mask,
                topic_one_hot, author_id, job_id, location_id, affiliation_id, history_feature):

        # Encode the news representation using BERT
        # seq_len is replaced by attention_mask
        outputs = self.bert(input_ids=inputs,
                            token_type_ids=segment_types,
                            attention_mask=attention_mask)

        # We use the pooler_output, which corresponds to the [CLS] token
        cls_encoding = outputs.pooler_output

        #dataset specific features:
        topic_fea = self.topic_embed(topic_one_hot)
        author_fea = self.author_embed(author_id)
        job_fea = self.job_embed(job_id)
        location_fea = self.location_embed(location_id)
        affiliation_fea = self.affiliation_embed(affiliation_id)

        # Concat author-related features
        author_features_combined = torch.cat((topic_fea, author_fea, job_fea, location_fea, affiliation_fea), dim=-1)
        author_feature = self.author_feature_map(author_features_combined)

        history_feature = self.history_feature_map(history_feature)

        # Concat all features for final classification
        combined_features = torch.cat((cls_encoding, author_feature, history_feature), dim=-1)

        return self.classifier(combined_features)
    
def transform_fn(text, topic_one_encoding, author_id, job_id, location_id, affiliation_id, history_feature, venue_ids, label):
    max_len = 256

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # Adds [CLS] and [SEP]
        max_length=max_len,
        truncation=True,
        padding=False,            # We will pad in the collate_fn
        return_token_type_ids=True
    )

    data = np.array(encoding['input_ids'], dtype='int64')
    length = np.array(len(data), dtype='int64') # This is now just the length, not a tensor
    segment_type = np.array(encoding['token_type_ids'], dtype='int64')

    # history_feature is already a numpy array, just ensure type
    history_feature = np.array(history_feature, dtype=np.float32)

    # Return all features as standard types for batching
    return (data, length, segment_type, topic_one_encoding, author_id, job_id,
            location_id, affiliation_id, history_feature, label)

def collate_fn(batch):
    (data, lengths, segment_type, topic_one_encoding, author_id, job_id,
     location_id, affiliation_id, history_feature, label) = zip(*batch)

    # Pad sequences
    data_padded = pad_sequence(
        [torch.as_tensor(d, dtype=torch.long) for d in data],
        batch_first=True, padding_value=padding_id
    )

    seg_types_padded = pad_sequence(
        [torch.as_tensor(s, dtype=torch.long) for s in segment_type],
        batch_first=True, padding_value=0
    )

    attention_mask = (data_padded != padding_id).long()

    # Convert to tensors (no stacking needed if already numbers)
    lengths = torch.tensor([int(l) for l in lengths], dtype=torch.long)
    topic_one_encoding = torch.as_tensor(topic_one_encoding, dtype=torch.float32)
    author_id = torch.as_tensor(author_id, dtype=torch.long)
    job_id = torch.as_tensor(job_id, dtype=torch.long)
    location_id = torch.as_tensor(location_id, dtype=torch.long)
    affiliation_id = torch.as_tensor(affiliation_id, dtype=torch.long)
    history_feature = torch.as_tensor(history_feature, dtype=torch.float32)
    label = torch.as_tensor(label, dtype=torch.long)

    # Optional pinning (usually safe to skip unless you explicitly use pin_memory=True in DataLoader)
    for t in [data_padded, seg_types_padded, attention_mask, lengths, topic_one_encoding,
              author_id, job_id, location_id, affiliation_id, history_feature, label]:
        t = t.pin_memory()

    return (data_padded, seg_types_padded, attention_mask,
            topic_one_encoding, author_id, job_id, location_id,
            affiliation_id, history_feature, label)