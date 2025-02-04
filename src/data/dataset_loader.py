import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from data.preprocess import extract_matrice_embedding
from utils.utils import get_models_and_tokenizers
from config_loader import ConfigLoader
from transformers import DataCollatorWithPadding
from utils.utils import model_inference
from data.preprocess import extract_matrice_embedding

class PretrainDataset(Dataset):
    def __init__(self, dataset, labels = None):
        super().__init__()
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset["english_text_input_ids"])

    def __getitem__(self, index):
        english_text_input_ids = torch.tensor(self.dataset["english_text_input_ids"][index], dtype = torch.long)
        english_text_attention_mask = torch.tensor(self.dataset["english_text_attention_mask"][index], dtype = torch.long)
        vietnamese_text_input_ids  = torch.tensor(self.dataset["vietnamese_text_input_ids"][index], dtype = torch.long)
        
        if self.labels is not None:
            labels = torch.tensor(self.labels[index], dtype=torch.long)
            return {
                "english_text_input_ids": english_text_input_ids,
                "vietnamese_text_input_ids": vietnamese_text_input_ids,
                "english_text_attention_mask":english_text_attention_mask,
                "labels": labels
            }

        return {
            "english_text_input_ids": english_text_input_ids,
            "vietnamese_text_input_ids": vietnamese_text_input_ids,
            "english_text_attention_mask":english_text_attention_mask
        }
    
def load_dataset(file_path: str, list_of_columns: list = None,
                rename_columns: dict = None          
    ) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    if list_of_columns is not None:
        df = df[list_of_columns]

    if rename_columns is not None:
        df = df.rename(columns=rename_columns)
    return df

def random_balance_pretrain_dataset(data: pd.DataFrame, neu_dataset_size = 50000, neg_dataset_size = 50000, pos_dataset_size = 50000, random_state = 42) -> pd.DataFrame:
    """
    Get random data from the dataset (the number must be balance).
    Args:
        data (pd.DataFrame): The dataset to sample from.
        dataset_size (int): The number of samples to get.
    Returns:
        pd.DataFrame: The sampled dataset.
    """
    neu_data = data[data['Label'] == 1].sample(neu_dataset_size, random_state=random_state)
    neg_data = data[data['Label'] == 0].sample(neg_dataset_size, random_state=random_state)
    pos_data = data[data['Label'] == 2].sample(pos_dataset_size, random_state=random_state)
    return pd.concat([neu_data, neg_data, pos_data]).reset_index(drop=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

def pretrain_collate_fn(batch):
    #print("Batch")
    english_model, vietnamese_model, english_tokenizer, vietnamese_tokenizer = get_models_and_tokenizers()

    english_text_input_ids = [sample["english_text_input_ids"] for sample in batch]
    vietnamese_text_input_ids  = [sample["vietnamese_text_input_ids"] for sample in batch]
    english_text_attention_mask = [sample["english_text_attention_mask"] for sample in batch]

    eng_padding_collator = DataCollatorWithPadding(tokenizer=english_tokenizer, padding=True, return_tensors='pt')
    vi_padding_collator = DataCollatorWithPadding(tokenizer=vietnamese_tokenizer, padding=True, return_tensors='pt')

    english_batch = eng_padding_collator({'input_ids': english_text_input_ids, "attention_mask" : english_text_attention_mask})
    vietnamese_batch = vi_padding_collator({'input_ids': vietnamese_text_input_ids})

    vietnamese_embedding = extract_matrice_embedding(vietnamese_batch['input_ids'], vietnamese_model)
    english_logits, english_prediction = model_inference(english_model, english_batch['input_ids'], english_batch['attention_mask'])

    return {
        'vietnamese_text_embeddings': vietnamese_embedding,
        'english_logits' : english_logits,
        'english_prediction' : english_prediction,
        'english_ids' : english_batch['input_ids'] ,
        'vi_ids' : vietnamese_batch['input_ids']
    }


def split_dataset(dataset, train_size, dev_size):
    train_size = int(train_size * len(dataset))  
    dev_size = int(dev_size * len(dataset))    
    test_size = len(dataset) - train_size - dev_size 
    print(train_size, dev_size, test_size)
    torch.manual_seed(42)
    train_dataset, dev_dataset, test_dataset = random_split(dataset, [train_size, dev_size, test_size])
    print(dev_dataset[0])
    return train_dataset, dev_dataset, test_dataset

