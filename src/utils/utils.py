import torch.nn as nn
from config_loader import ConfigLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import transformers
import torch 

english_model = None
vietnamese_model = None
englist_tokenizer = None
vietnamese_tokenizer = None
id2label = {0: "NEGATIVE", 1 : "NEUTRAL" , 2: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 2 , "NEUTRAL" : 1}

def load_vietnamese_model():
    global vietnamese_model
    if vietnamese_model is None:
        vietnamese_model = AutoModel.from_pretrained(ConfigLoader.get("model.vietnamese_model"))
    return vietnamese_model.to(ConfigLoader.get("device"))

def load_english_model():
    global english_model
    if english_model is None:
        english_model = AutoModelForSequenceClassification.from_pretrained(ConfigLoader.get("model.english_model"), id2label=id2label, label2id=label2id)
    return english_model.to(ConfigLoader.get("device"))

def load_vietnamese_tokenizer():
    global vietnamese_tokenizer
    if vietnamese_tokenizer is None:
        vietnamese_tokenizer = AutoTokenizer.from_pretrained(ConfigLoader.get("model.vietnamese_model"))
    return vietnamese_tokenizer

def load_english_tokenizer():
    global englist_tokenizer
    if englist_tokenizer is None:
        englist_tokenizer = AutoTokenizer.from_pretrained(ConfigLoader.get("model.english_model"))
    return englist_tokenizer

def get_vietnamese_model():
    global vietnamese_model
    if vietnamese_model is None:
        load_vietnamese_model()
    return vietnamese_model

def get_english_model():
    global english_model
    if english_model is None:
        load_english_model()
    return english_model

def get_vietnamese_tokenizer():
    global vietnamese_tokenizer
    if vietnamese_tokenizer is None:
        load_vietnamese_tokenizer()
    return vietnamese_tokenizer

def get_english_tokenizer():
    global englist_tokenizer
    if englist_tokenizer is None:
        load_english_tokenizer()
    return englist_tokenizer

def get_models_and_tokenizers():
    return get_english_model(), get_vietnamese_model(), get_english_tokenizer(), get_vietnamese_tokenizer()
    
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def count_parameters(model):
    """
    Count the total number of parameters in a PyTorch model.
    
    Args:
        model (nn.Module): The PyTorch model instance.
        
    Returns:
        int: The total number of parameters.
        int: The number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def model_inference(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        attention_mask = attention_mask.to(model.device)
        input_ids = input_ids.to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1)

    return logits, prediction