import pandas as pd
from pyvi.ViTokenizer import tokenize
import torch
import transformers
from torch.utils.data import random_split

from config_loader import ConfigLoader


def convert_label(dataset_df: pd.DataFrame, current_negative, current_neutral, current_positive):
    dataset_df['Label'] = dataset_df['Label'].apply(lambda x: 0 if x == current_negative else 1 if x == current_neutral else 2 if x == current_positive else x)
    return dataset_df

# Tokenize sentence
def tokenize_text(english_text, vietnamese_text, vietnamese_tokenizer, english_tokenizer, seed=42):
    # Tokenize English sentences and generated positives/negatives
    english_text_tokenized = english_tokenizer(english_text, padding="max_length", max_length=ConfigLoader.get("dataset.padding_max_length"), truncation=True)
    # Tokenize Vietnamese sentences
    vietnamese_text_segmented = [tokenize(sentence) for sentence in vietnamese_text]
    vietnamese_text_tokenized = vietnamese_tokenizer(vietnamese_text_segmented, padding="max_length", max_length=ConfigLoader.get("dataset.padding_max_length"), truncation=True)
    
    return {
        "english_text_input_ids": english_text_tokenized["input_ids"],
        "english_text_attention_mask": english_text_tokenized["attention_mask"],
        "vietnamese_text_input_ids": vietnamese_text_tokenized["input_ids"], 
    }

def extract_matrice_embedding(input_ids, model: transformers.PreTrainedModel):
    # Get the first embedding layer of the model (check for bert, roberta, etc.)
    embedding_layer = model.get_input_embeddings()
    # Extract the embeddings of the input_ids
    embeddings = embedding_layer(input_ids.to(model.device))    
    return embeddings.to("cpu")



