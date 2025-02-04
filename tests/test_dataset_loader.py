from src.data.dataset_loader import load_dataset, random_balance_pretrain_dataset, PretrainDataset, pretrain_collate_fn, get_dataloader
from src.data.preprocess import tokenize_text
from src.utils.utils import get_vietnamese_tokenizer, get_english_tokenizer
from src.config_loader import ConfigLoader
from torch.utils.data import DataLoader
import os

ConfigLoader.load_config(os.path.join(os.path.dirname(__file__), "../config/config.yaml"))
print(ConfigLoader.get("training.batch_size"))
path = os.path.join(os.path.dirname(__file__), "../data/pretrain_data/phomt.csv")
phomt_df = random_balance_pretrain_dataset(load_dataset(path))
tokenized_phomt_data = tokenize_text(phomt_df["EnglishText"].tolist(), phomt_df["VietnameseText"].tolist(), get_vietnamese_tokenizer(), get_english_tokenizer())
pretrain_dataset = PretrainDataset(tokenized_phomt_data)

pretrain_data_loader = DataLoader(pretrain_dataset, batch_size=ConfigLoader.get("training.batch_size"), collate_fn=pretrain_collate_fn)
batch_0 = next(iter(pretrain_data_loader))
print(batch_0)
