# %%[markdown]
# Train a toxicity probe using two vectors
# This is a more complex probe that uses the mean and last token of the input sequence
# instead of just the last token.
# Results are not so clear as we saw with the single vector probe. 

# | VECTOR         | TOP TOKENS                                                                    |
# |:---------------|:------------------------------------------------------------------------------|
# | MLP.v_993^3    | ette, antine, tics, stad, Zur, onne                                           |
# | MLP.v_2912^13  | closest, eways, cade, herd, line, brush                                       |
# | MLP.v_1479^0   | scrolls, Orb, Bulgaria, dylib, Petra, swer                                    |
# | MLP.v_950^1    | assis, Software, Hass, ISO, Tip, Widget                                       |
# | MLP.v_3448^2   | Copyright, ¶, declass, Explan, EFF, Prosecut                                  |
# | MLP.v_3844^5   | inka, alty, anmar, lihood, opp, Shogun                                        |
# | MLP.v_304^23   | sw, per, body, plain, -, LE                                                   |
# | MLP.v_1554^1   | chool, rict, uzzle, ICT, antically, icz                                       |
# | MLP.v_607^2    | aird, cast, rede, outing, Keller, chet                                        |
# | MLP.v_1617^11  | .}, Corpus, answer, Guer, Tags, pacif                                         |
# | MLP.v_3712^4   | PS, PLA, KP, HM, SA, KS                                                       |
# | MLP.v_3847^7   | ordes, helm, Cla, Germ, ignant, wig                                           |
# | MLP.v_1289^18  | angles, angle, perpendicular, collisions, acceleration, piv                   |
# | MLP.v_2301^6   | 龍喚士, vity, Application, posure, Activity, 龍契士                           |
# | MLP.v_2082^16  | theorem, algebra, calculus, equations, mathematical, arithmetic               |
# | SVD.U_Toxic[0] | covari, constants, lambda, arithmetic, theorem, algebra                       |
# | SVD.U_Toxic[1] | 76561, DragonMagazine, サーティワン, Buyable, cloneembedreportprint, ÃÂÃÂÃÂÃÂ |
# | SVD.U_Toxic[2] | 1996, 1986, 1999, 1991, 1989, 1989                                            |


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# %%
class ToxicityDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=1024):
        self.encodings = tokenizer(comments, return_tensors='pt', padding=True,
                                 max_length=max_length, truncation=True, return_attention_mask=True)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'tokens': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        
# %%



class ToxicityProbe(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # First combine the concatenated features back to hidden_size
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        # Then project to classification logits
        self.classifier = nn.Linear(hidden_size, 2)
        
    def forward(self, hidden_states, attention_mask):
        # Get the last token position for each sequence in the batch
        last_token_pos = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        
        # Get hidden states at the last real token for each sequence
        last_token_hidden = hidden_states[torch.arange(batch_size), last_token_pos]
        
        # Get mean of all non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_hidden = hidden_states * mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        count_tokens = attention_mask.sum(dim=1, keepdim=True)
        averaged_hidden = sum_hidden / count_tokens
        
        # Concatenate mean and last token features
        combined_features = torch.cat([averaged_hidden, last_token_hidden], dim=1)
        
        # Project back to model's hidden dimension
        hidden = self.combine(combined_features)
        # Get classification logits
        logits = self.classifier(hidden)
        return logits

# %%

def evaluate_probe(model, probe, val_loader, device):
    probe.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Final Evaluation'):
            tokens = batch['tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name == f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
            )
            hidden_states = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"]
            logits = probe(hidden_states, attention_mask)
            
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-toxic', 'Toxic']))
    
    # Log confusion matrix to wandb
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=['Non-toxic', 'Toxic'])
    })

def train_toxicity_probe(model, probe, train_loader, val_loader, device, 
                        num_epochs=3, learning_rate=1e-3):
    # Initialize wandb
    wandb.init(
        project="toxicity-probe",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "model": model.cfg.model_name,
        }
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        probe.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training loop with tqdm
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            tokens = batch['tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name == f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
                )
                hidden_states = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"]
            
            # Forward pass through probe with attention mask
            logits = probe(hidden_states, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar description with current loss and accuracy
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(predicted == labels).float().mean().item():.3f}'
            })
            
            # Log training metrics per batch
            wandb.log({
                "batch_loss": loss.item(),
                "batch_accuracy": (predicted == labels).float().mean().item(),
                "batch": epoch * len(train_loader) + batch_idx
            })
        
        train_accuracy = 100 * correct/total
        avg_train_loss = total_loss/len(train_loader)
        
        # Validation loop with tqdm
        probe.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                       leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                tokens = batch['tokens'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name == f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
                )
                hidden_states = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"]
                logits = probe(hidden_states, attention_mask)
                
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Calculate validation loss
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Update progress bar description
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(predicted == labels).float().mean().item():.3f}'
                })
        
        val_accuracy = 100 * val_correct/val_total
        avg_val_loss = val_loss/len(val_loader)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
        })
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
    
    # After training is complete, run evaluation
    print("\nFinal Evaluation:")
    evaluate_probe(model, probe, val_loader, device)
    
    # Close wandb run
    wandb.finish()

# %%

def main():
    # %%
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load GPT-2 medium model using TransformerLens
    model = HookedTransformer.from_pretrained(
        "gpt2-medium",
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=device
    )
    model.eval()

# %%
    
    # Load and preprocess the Jigsaw dataset
    df = pd.read_csv('data/train.csv')  # Adjust filename as needed
    comments = df['comment_text'].tolist()
    labels = df['toxic'].tolist()


# %%
    
    # Split the dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        comments, labels, test_size=0.1, random_state=42
    )
    
    # %%
    
    # Create datasets using the model's tokenizer
    train_dataset = ToxicityDataset(train_texts, train_labels, model.tokenizer)
    val_dataset = ToxicityDataset(val_texts, val_labels, model.tokenizer)
    
    # %%
    # Show first 5 elements of training dataset
    print("First 5 elements of training dataset:")
    for i in range(5):
        print(train_dataset[i])
        print(train_dataset[i]['tokens'].shape)
    # %%
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # %%
    
    # Initialize probe
    probe = ToxicityProbe(model.cfg.d_model)  # Use model's hidden size
    probe.to(device)

    # %%
    
    # Train the probe
    train_toxicity_probe(model, probe, train_loader, val_loader, device, num_epochs=2)
    
    # %%
    
    # Save the trained probe
    torch.save(probe.state_dict(), 'toxicity_probe.pt')

    # Example of using the probe for inference
    def predict_toxicity(text):
        model.eval()
        probe.eval()
        with torch.no_grad():
            tokens = model.tokenizer.tokenize([text], return_tensors='pt').to(device)
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name == f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
            )
            hidden_states = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"]
            logits = probe(hidden_states)
            probs = torch.softmax(logits, dim=1)
            return probs[0][1].item()  # Probability of toxic class
# %%
if __name__ == '__main__':
    main()
# %%
