import torch
from tqdm.auto import tqdm
import os

from config_loader import ConfigLoader


def train_mapper(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs, n_epochs_stop, save_path, english_model = None):
    """
    Train the Mapper model with early stopping and auto-saving.

    Args:
        model: The Mapper model to train.
        optimizer: Optimizer for training.
        criterion: Loss function.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        num_epochs (int): Maximum number of epochs.
        english_model: Pretrained frozen English model.
        n_epochs_stop (int): Number of epochs to wait for improvement before stopping.
        save_path (str): Path to save the best model.
    """
    for param in english_model.parameters():
        param.requires_grad = False

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()

        train_losses = 0.0
        val_losses = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            vietnamese_text_embeddings = batch["vietnamese_text_embeddings"].to(ConfigLoader.get("device"))
            english_logits = batch["english_logits"].to(ConfigLoader.get("device"))
            english_prediction = batch["english_prediction"].to(ConfigLoader.get("device"))

            mapped_embeddings = model(vietnamese_text_embeddings)
            fake_english_last_hidden_state = english_model.roberta.encoder(mapped_embeddings)[0]
            fake_english_logits = english_model.classifier(fake_english_last_hidden_state)

            if ConfigLoader.get("training.loss") == "cross_entropy":
                loss = criterion(fake_english_logits, english_prediction)
            elif ConfigLoader.get("training.loss") == "mse":
                loss = criterion(fake_english_logits, english_logits)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses += loss.item()
        avg_train_loss = train_losses / len(train_dataloader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                vietnamese_text_embeddings = batch["vietnamese_text_embeddings"].to(ConfigLoader.get("device"))
                english_logits = batch["english_logits"].to(ConfigLoader.get("device"))
                english_prediction = batch["english_prediction"].to(ConfigLoader.get("device"))

                mapped_embeddings = model(vietnamese_text_embeddings)
                fake_english_last_hidden_state = english_model.roberta.encoder(mapped_embeddings)[0]
                fake_english_logits = english_model.classifier(fake_english_last_hidden_state)

                if ConfigLoader.get("training.loss") == "cross_entropy":
                    loss = criterion(fake_english_logits, english_prediction)
                elif ConfigLoader.get("training.loss") == "mse":
                    loss = criterion(fake_english_logits, english_logits)

                total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Training Loss: {avg_train_loss}")
            print(f"Validation Loss: {avg_val_loss}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
                print("Model saved with val loss:", best_val_loss)
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    print("Early stopping!")
                    break

    # Load the best model
    model.load_state_dict(torch.load(save_path))
    return model



