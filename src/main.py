import cmd
from model.mapper import MapperModel
from model.train_mapper import train_mapper
from data.dataset_loader import load_dataset, random_balance_pretrain_dataset, pretrain_collate_fn, split_dataset, PretrainDataset
from config_loader import ConfigLoader
from torch.utils.data import DataLoader
from data.preprocess import tokenize_text
from utils.utils import get_models_and_tokenizers, get_english_model, get_english_tokenizer, get_vietnamese_tokenizer, get_vietnamese_model, model_inference
import torch
from model.evaluation import evaluate_translate, evaluate_mapper, eval_translate_collate_fn
from collections import Counter
import os
from tqdm.auto import tqdm

# Global variables
dataset = None
train_dataset = val_dataset = test_dataset = None
pretrain_data_loader = val_data_loader = None
mapper_model = None
english_model = vietnamese_model = None
english_tokenizer = vietnamese_tokenizer = None

english_model, vietnamese_model, english_tokenizer, vietnamese_tokenizer = get_models_and_tokenizers()

class MapperCLI(cmd.Cmd):
    intro = "Welcome to the Mapper CLI. Type help or ? to list commands."
    prompt = "(mapper) "

    def __init__(self):
        super().__init__()
        self.step_flags = {
            "load_pretrain_dataset": False,
            "random_balance_pretrain_dataset": False,
            "tokenized_dataset": False,
            "convert_to_custom_dataset": False,
            "split_dataset": False,
            "get_data_loader": False,
            "train": False,
        }

    def do_load_dataset(self, arg):
        """
        \nLoad the dataset: load_dataset -p <file_path>\n]
        \n-p <file_path>: The path to the dataset file (optional). Default value: defined on config file.
        """
        global dataset
        args = arg.split()
        file_path = None

        if "-p" in args:
            try:
                file_path = args[args.index("-p") + 1]
            except IndexError:
                print("Error: No file path provided")
            return
        
        if file_path is None:
            file_path = ConfigLoader.get("model.save_path")
        dataset = load_dataset(ConfigLoader.get("dataset.pretrain_file_path"),
                               list_of_columns=["EnglishSentences", "VietnameseSentences", "label"],
                               rename_columns={"EnglishSentences": "EnglishText", "VietnameseSentences": "VietnameseText", "label": "Label"})
        self.step_flags["load_pretrain_dataset"] = True
        print("Dataset loaded successfully.")
        print(dataset.head(10))

    def do_balance(self, arg):
        """
        \nBalance the dataset: balance
        """
        global dataset
        if not self.step_flags["load_pretrain_dataset"]:
            print("Error: You must load the dataset first.")
            return
        dataset = random_balance_pretrain_dataset(
            dataset, neu_dataset_size=50000, neg_dataset_size=50000, pos_dataset_size=50000
        )
        self.step_flags["random_balance_pretrain_dataset"] = True
        print("Dataset balanced successfully.")
        print(dataset.head(10))
        print(dataset["Label"].value_counts())

    def do_tokenize(self, arg):
        "\nTokenize the dataset: tokenize"
        global dataset
        if not self.step_flags["random_balance_pretrain_dataset"]:
            print("Error: You must balance the dataset first.")
            return
        print(dataset.head(3))
        dataset = tokenize_text(
            dataset["EnglishText"].tolist(), dataset["VietnameseText"].tolist(),
            vietnamese_tokenizer, english_tokenizer
        )
        self.step_flags["tokenized_dataset"] = True
        print("Dataset tokenized successfully.")

    def do_convert(self, arg):
        "\nConvert to custom dataset: convert"
        global dataset
        if not self.step_flags["load_pretrain_dataset"]:
            print("Error: You must load the dataset first.")
            return
        dataset = PretrainDataset(dataset=dataset)
        self.step_flags["convert_to_custom_dataset"] = True
        print("Dataset converted successfully.")
        print("Dataset length:", len(dataset))

    def do_split(self, arg):
        """
        \nSplit the dataset: split -train_size <train_size> -val_size <val_size> 
        \n-train_size: The size of the training set (optional). Default value: defined on config file. 
        \n-val_size: The size of the validation set (optional). Default value: defined on config file. 
        """
        args = arg.split()
        train_size = None,
        val_size = None

        if "-train_size" and "-val_size" in args:
            try:
                train_size = args[args.index("-train_size") + 1]
                val_size = args[args.index("-val_size") + 1]
            except IndexError:
                print("Error: No train and val size provided")
            return
        if ("-train_size" in args) ^ ("-val_size" in args):
            print("Error: You must provide both train and val size or none of them.")
            return
        else:
            train_size = ConfigLoader.get("dataset.train_size")
            val_size = ConfigLoader.get("dataset.val_size")

        global train_dataset, val_dataset, test_dataset
        if not self.step_flags["random_balance_pretrain_dataset"]:
            print("Error: You must balance the dataset first.")
            return
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset, train_size, val_size
        )
        self.step_flags["split_dataset"] = True
        print("Dataset split successfully.")
        print("Lengths:", len(train_dataset), len(val_dataset))

    def do_dataloader(self, arg):
        """
        \nCreate data loaders: dataloader -batch_size <batch_size> 
        \n-batch_size: The batch size (optional). Default value: defined on config file.
        """
        args = arg.split()
        batch_size = None
        if "-batch_size" in args:
            try:
                batch_size = args[args.index("-batch_size") + 1]
            except IndexError:
                print("Error: No file path provided")
            return
        else:
            batch_size = ConfigLoader.get("training.batch_size")
        
        global train_dataset, val_dataset, pretrain_data_loader, val_data_loader
        if not self.step_flags["split_dataset"]:
            print("Error: You must split the dataset first.")
            return
        pretrain_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=pretrain_collate_fn
        )
        val_data_loader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=pretrain_collate_fn
        )

        next(iter(pretrain_data_loader))
        print("First batch from val_data_loader:", next(iter(val_data_loader)))
        print("Data loader lengths:", len(pretrain_data_loader), len(val_data_loader))
        self.step_flags["get_data_loader"] = True
        print("Data loaders created successfully.")

    def do_train(self, arg):
        """
        \nTrain the model: train
        \nThis command will train the model using the data loaders created in the previous step. If not created, it will create them.
        """
        global pretrain_data_loader, val_data_loader, mapper_model, english_model, vietnamese_model

        for step in self.step_flags.keys():
            if not self.step_flags[step]:
                if "load" in step and "loader" not in step:
                    self.do_load_dataset(arg)
                elif "balance" in step:
                    self.do_balance(arg)
                elif "tokenize" in step:
                    self.do_tokenize(arg)
                elif "convert" in step:
                    self.do_convert(arg)
                elif "split" in step:
                    self.do_split(arg)
                elif "loader" in step:
                    self.do_dataloader(arg)

        # Load models
        english_model, vietnamese_model, english_tokenizer, vietnamese_tokenizer = get_models_and_tokenizers()
        mapper_model = MapperModel(
            in_embedding_dim=ConfigLoader.get("model.in_embedding_dim"), 
            out_embedding_dim=ConfigLoader.get("model.out_embedding_dim")
        ).to(ConfigLoader.get("device"))

        optimizer = torch.optim.Adam(
            mapper_model.parameters(), lr=ConfigLoader.get("training.learning_rate"),
            weight_decay=ConfigLoader.get("training.weight_decay")
        )

        criterion = torch.nn.CrossEntropyLoss() if ConfigLoader.get("training.loss") == "cross_entropy" else torch.nn.MSELoss()
        # Train model
        train_mapper(
            model=mapper_model, optimizer=optimizer, criterion=criterion,
            train_dataloader=pretrain_data_loader, val_dataloader=val_data_loader,
            english_model=english_model, num_epochs=ConfigLoader.get("training.num_epochs"),
            n_epochs_stop=ConfigLoader.get("training.n_epochs_stop"),
            save_path=ConfigLoader.get("model.save_path")
        )
        self.step_flags["train"] = True
        print("Training completed successfully.")



    def do_eval_translate(self, arg):
        "\nEvaluate translation: eval_translate"
        evaluate_translate()
    
    def do_eval_mapper(self, arg):
        """
        \nEvaluate mapper: eval_mapper -p <file_path>
        \n-p: Path to the mapper file to be evaluated.
        """
        args = arg.split()
        file_path = None

        if "-p" in args:
            try:
                file_path = args[args.index("-p") + 1]
            except IndexError:
                print("Error: No file path provided")
            return
        
        if file_path is None:
            file_path = ConfigLoader.get("model.save_path")
        
        evaluate_mapper(file_path=file_path)

    def do_label_data(self, arg):
        """\nLabel for sentiment analysis task for the dataset: label -p <file_path> -s <save_path>
        -p: Path to the dataset file.
        -s: Path to save the labeled dataset.
        """
        args = arg.split()
        file_path = None
        save_path = None

        print("Here")
        if "-p" in args:
            try:
                file_path = args[args.index("-p") + 1]
                print(file_path)
            except IndexError:
                print("Error: No file path provided")
                return
        
        if "-s" in args:
            try:
                save_path = args[args.index("-s") + 1]
                print(save_path)
            except IndexError:
                print("Error: No save path provided")
                return
        
        print("Here 2")
        if file_path is None:
            file_path = ConfigLoader.get("dataset.pretrain_file_path")
        if save_path is None:
            save_path = ConfigLoader.get("dataset.pretrain_data_dir")
        
        print("Here 3")
        dataset = load_dataset(file_path, list_of_columns=["VietnameseText", "EnglishText" , "Label"])

        tokenized_dataset = tokenize_text(
            dataset["EnglishText"].tolist(), dataset["VietnameseText"].tolist(),
            get_vietnamese_tokenizer(), get_english_tokenizer())
        
        custom_dataset = PretrainDataset(dataset=tokenized_dataset, labels=dataset["Label"].tolist())
        dataloader = DataLoader(custom_dataset, batch_size=1024, collate_fn=eval_translate_collate_fn)

        list_labels = []
        list_neg_probs = []
        list_neu_probs = []
        list_pos_probs = []

        for batch in tqdm(dataloader):
            with torch.no_grad():
                logits, predictions = batch["english_logits"].to("cpu"), batch["english_prediction"].to("cpu")
                list_labels += predictions.tolist()
                logits = torch.softmax(logits, dim=1)
                neg_prob, neu_prob, pos_prob = logits[:, 0], logits[:, 1], logits[:, 2]
                list_neg_probs += neg_prob.tolist()
                list_neu_probs += neu_prob.tolist()
                list_pos_probs += pos_prob.tolist()

        dataset["NegProb"] = list_neg_probs
        dataset["NeuProb"] = list_neu_probs
        dataset["PosProb"] = list_pos_probs
        dataset["Label"] = list_labels
        dataset.to_csv(save_path, index=False)

        print("Dataset labeled successfully.")
    
    def do_help(self, arg):
        "\nHelp: help"
        return super().do_help(arg)

    def do_exit(self, arg):
        "\nExit the CLI: exit"
        print("Exiting CLI. Goodbye!")
        return True

    def do_clear(self, arg):
        "\nClear the screen: clear"
        os.system("cls" if os.name == "nt" else "clear")
if __name__ == "__main__":
    MapperCLI().cmdloop()
