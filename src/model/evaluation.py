import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from utils.utils import get_english_model, get_english_tokenizer, get_vietnamese_tokenizer, get_vietnamese_model
from data.dataset_loader import PretrainDataset, pretrain_collate_fn
from data.preprocess import tokenize_text, extract_matrice_embedding
from transformers import DataCollatorWithPadding
from config_loader import ConfigLoader
from tqdm.auto import tqdm
from utils.utils import model_inference
from sklearn.metrics import classification_report
from model.mapper import MapperModel

def evaluate_translate():
    english_model = get_english_model()
    english_model.eval()

    list_test_datasets = ["aivivn", "vlsp", "ntcscv", "viocd"]
    translated_file_path = ConfigLoader.get("dataset.translated_file_path")

    for test_dataset in list_test_datasets:

        dataset = pd.read_csv(translated_file_path + "test_" + test_dataset + ".csv")
        labels = dataset["Label"].tolist()
        dataset = tokenize_text(dataset["EnglishText"].tolist(),
                                dataset["EnglishText"].tolist(),
                                get_vietnamese_tokenizer(),
                                get_english_tokenizer())
        dataset = PretrainDataset(dataset)
        dataloader = DataLoader(dataset, batch_size=1024, collate_fn=eval_translate_collate_fn)

        list_english_predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                english_logits, english_prediction = batch["english_logits"].to("cpu"), \
                                                    batch["english_prediction"].to("cpu")
                
                # if test_dataset in ["aivivn", "ntcscv", "viocd"]:
                #     english_prediction = torch.argmax(english_logits[:, [0, 2]], dim=1) * 2
                
                list_english_predictions += english_prediction.tolist()
            print("="*10)
            print(f"Evaluation result on {test_dataset}:")
            print(classification_report(labels, list_english_predictions))
    

def eval_translate_collate_fn(batch):
    #print("Batch")
    english_model, english_tokenizer = get_english_model(), get_english_tokenizer()

    english_text_input_ids = [sample["english_text_input_ids"] for sample in batch]
    english_text_attention_mask = [sample["english_text_attention_mask"] for sample in batch]

    eng_padding_collator = DataCollatorWithPadding(tokenizer=english_tokenizer, padding=True, return_tensors='pt')
    english_batch = eng_padding_collator({'input_ids': english_text_input_ids, "attention_mask" : english_text_attention_mask})
    english_logits, english_prediction = model_inference(english_model, english_batch['input_ids'], english_batch['attention_mask'])

    return {
        'english_logits' : english_logits,
        'english_prediction' : english_prediction,
    }

def eval_mapper_collate_fn(batch):
    vietnamese_tokenizer, vietnamese_model = get_vietnamese_tokenizer(), get_vietnamese_model()

    vietnamese_text_input_ids = [sample["vietnamese_text_input_ids"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    labels = torch.tensor(labels, dtype=torch.long)

    vi_padding_collator = DataCollatorWithPadding(tokenizer=vietnamese_tokenizer, padding=True, return_tensors='pt')
    vietnamese_batch = vi_padding_collator({'input_ids': vietnamese_text_input_ids})

    vietnamese_embedding = extract_matrice_embedding(vietnamese_batch["input_ids"], vietnamese_model)

    return {
        "vietnamese_text_embeddings": vietnamese_embedding,
        "labels": labels
    }

def evaluate_mapper(file_path = ConfigLoader.get("model.save_path")):
    english_model = get_english_model()
    vietnamese_model = get_vietnamese_model()
    english_model.eval()
    vietnamese_model.eval()

    model = MapperModel()

    model.load_state_dict(torch.load(file_path))
    model = model.to(ConfigLoader.get("device"))
    
    list_test_datasets = ["aivivn", "vlsp", "ntcscv", "viocd"]
    translated_file_path = ConfigLoader.get("dataset.translated_file_path")

    for test_dataset in list_test_datasets:
        dataset = pd.read_csv(translated_file_path + "test_" + test_dataset + ".csv")
        labels = dataset["Label"].tolist()
        dataset = tokenize_text(dataset["EnglishText"].tolist(),
                                dataset["VietnameseText"].tolist(),
                                get_vietnamese_tokenizer(),
                                get_english_tokenizer())
        dataset = PretrainDataset(dataset, labels)
        dataloader = DataLoader(dataset, batch_size=1024, collate_fn=eval_mapper_collate_fn)

        with torch.no_grad():
            all_predictions = []
            all_labels = []
            for batch in tqdm(dataloader):
                batch_vietnamese_text_embeddings, batch_labels = batch["vietnamese_text_embeddings"].to(ConfigLoader.get("device")),\
                                                                 batch["labels"].to(ConfigLoader.get("device"))
                mapped_embeddings = model(batch_vietnamese_text_embeddings)
                fake_english_last_hidden_state = english_model.roberta.encoder(mapped_embeddings)[0]
                fake_english_logits = english_model.classifier(fake_english_last_hidden_state)

                if test_dataset in ["aivivn", "ntcscv", "viocd"]:   
                    batch_predictions = torch.argmax(fake_english_logits[:, [0, 2]], dim=1) * 2
                else:
                    batch_predictions = torch.argmax(fake_english_logits, dim=1)
                batch_predictions = torch.argmax(fake_english_logits, dim=1)
                
                all_predictions += batch_predictions.tolist()
                all_labels += batch_labels.tolist()

            print("="*10)
            print(f"Evaluation result on {test_dataset}:")
            print(classification_report(labels, all_predictions))

                



                



    



