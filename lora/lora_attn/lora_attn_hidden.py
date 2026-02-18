import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from collections import Counter
import pandas as pd
import numpy as np
import wandb
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import seaborn as sns
import matplotlib.pyplot as plt

# Set all random seeds for full reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

set_seed(42)  # Initialize before anything else

# Initialize WandB with config tracking
wandb.init(project="lora", name="hidden")

# Configuration
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = "esmc_300m"
    label_csv = '/work/hdd/bdja/bpokhrel/mouse_proteome_with_sub_loc.csv'
    batch_size = 50
    num_epochs = 250
    patience = 20
    lr_esmc = 1e-5
    lr_classifier = 1e-4
    weight_decay = 1e-5
    max_len = 2000
    test_mode = False

    model_save_path = '/work/hdd/bdja/bpokhrel/lora/lora_attn/best_model_hidden.pth'

config = Config()
config.num_classes = len(pd.read_csv(config.label_csv)['subloc'].unique())
wandb.config.update(config.__dict__)  # Log all config parameters

# Initialize ESM Client with LoRA
client = ESMC.from_pretrained(config.model_name_or_path).to(config.device)
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["attn.layernorm_qkv.1", "attn.out_proj", "ffn.1", "ffn.3"],
)
client = get_peft_model(client, peft_config)

# Dataset Class
class ProteinDataset(Dataset):
    def __init__(self, label_csv, max_len=2000, test_mode=False):
        self.max_len = max_len
        df = pd.read_csv(label_csv)
        
        if test_mode:
            df = df.head(10)
            print("TEST MODE: Using only 10 samples")
        
        required_columns = {'UniProt_ID', 'Sequence', 'length', 'label'}
        if not required_columns.issubset(df.columns):
            raise KeyError(f"Missing columns: {required_columns - set(df.columns)}")
        
        df = df[df['length'] <= max_len]
        self.data = df[['UniProt_ID', 'length', 'Sequence', 'label']].values.tolist()
        self.sequences = [row[2] for row in self.data]
        self.labels = [row[3] for row in self.data]
        self.lengths = [row[1] for row in self.data]
        
        # Get unique sorted class names if available
        if 'label' in df.columns:
            self.classes = sorted(df['label'].unique().tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx])
        )

# Collate function
def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    return {
        "sequences": sequences,
        "labels": torch.stack(labels),
        "lengths": torch.stack(lengths)
    }

# Model Architecture
class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(x.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return self.layer_norm(context + x)

class DeepProteinClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.attention = Attention(embed_dim=960)
        self.fc_layers = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, mask=None):
        context = self.attention(x, mask)
        pooled = context.mean(dim=1)
        return self.fc_layers(pooled)

class ESMCClassifier(nn.Module):
    def __init__(self, esmc_model, classifier):
        super().__init__()
        self.esmc = esmc_model
        self.classifier = classifier
        
    def forward(self, batch):
        embeddings = []
        for seq in batch["sequences"]:
            protein = ESMProtein(sequence=seq)
            protein_tensor = self.esmc.encode(protein).to(config.device)
            outputs = self.esmc.logits(
                protein_tensor,
                LogitsConfig(return_embeddings=False,return_hidden_states=True )
            )
            hidden_s = outputs.hidden_states[1:]  # [29, 1, seq_len, embed_dim]
            hidden_s = hidden_s.squeeze(1)        # [29, seq_len, embed_dim]
            embed = torch.mean(hidden_s, dim=0)   # [seq_len, embed_dim]
            embeddings.append(embed)
        
        lengths = batch["lengths"].tolist()
        max_len = max(lengths)
        embed_dim = embeddings[0].shape[-1]
        batch_size = len(embeddings)
        
        padded_embeddings = torch.zeros((batch_size, max_len, embed_dim), device=config.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=config.device)
        
        for i, (embed, seq_len) in enumerate(zip(embeddings, lengths)):
            padded_embeddings[i, :seq_len] = embed[:seq_len]
            attention_mask[i, :seq_len] = 1
        
        return self.classifier(padded_embeddings, attention_mask)

# Metrics
def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
    return accuracy, f1

# In create_dataloaders() function, add these lines before returning:
def create_dataloaders():
    dataset = ProteinDataset(config.label_csv, test_mode=config.test_mode)
    labels = [item[3] for item in dataset.data]
    
    # Stratified split with fixed random state
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        stratify=labels,
        test_size=0.3,
        random_state=42
    )
    
    # Save validation indices to disk
    val_indices_path = os.path.join(os.path.dirname(config.model_save_path), 'val_indices.pt')
    torch.save({'val_idx': val_idx}, val_indices_path)
    wandb.save(val_indices_path)  # Upload to W&B
    
    # Rest of the function remains the same...
    train_data = sorted([dataset.data[i] for i in train_idx], key=lambda x: x[1])
    val_data = sorted([dataset.data[i] for i in val_idx], key=lambda x: x[1])
    
    train_subset = Subset(dataset, [dataset.data.index(item) for item in train_data])
    val_subset = Subset(dataset, [dataset.data.index(item) for item in val_data])
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=False,
    )
    
    return train_loader, val_loader

def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()
        }
    }, path)

def load_model(path, esmc_model, classifier):
    checkpoint = torch.load(path)
    model = ESMCClassifier(esmc_model, classifier).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path='confusion_matrix_hidden.png'):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def train_model():
    train_loader, val_loader = create_dataloaders()
    classifier = DeepProteinClassifier(config.num_classes).to(config.device)
    model = ESMCClassifier(client, classifier).to(config.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [
            {'params': model.esmc.parameters(), 'lr': config.lr_esmc},
            {'params': model.classifier.parameters(), 'lr': config.lr_classifier}
        ],
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config.num_epochs):
        if epochs_no_improve >= config.patience:
            print("Early stopping triggered")
            break
            
        # Training
        model.train()
        train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch["labels"].to(config.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            acc, f1 = calculate_metrics(outputs, batch["labels"].to(config.device))
            train_loss += loss.item()
            train_acc += acc
            train_f1 += f1
        
        # Validation
        model.eval()
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                outputs = model(batch)
                loss = criterion(outputs, batch["labels"].to(config.device))
                acc, f1 = calculate_metrics(outputs, batch["labels"].to(config.device))
                val_loss += loss.item()
                val_acc += acc
                val_f1 += f1
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_f1 /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, config.model_save_path)
            print(f"New best model saved to {config.model_save_path}")
        else:
            epochs_no_improve += 1
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr_esmc": optimizer.param_groups[0]['lr'],
            "lr_classifier": optimizer.param_groups[1]['lr']
        })
        
        print(f"Epoch {epoch+1}/{config.num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    
    wandb.finish()
    return model

def evaluate_model(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

if __name__ == "__main__":
    trained_model = train_model()
    val_indices_path = os.path.join(os.path.dirname(config.model_save_path), 'val_indices.pt')
    val_indices = torch.load(val_indices_path)['val_idx']
    
    # Load full dataset WITHOUT test_mode
    full_dataset = ProteinDataset(config.label_csv, test_mode=False)  # Changed this line
    
    # Create validation subset using the saved indices
    val_dataset = Subset(full_dataset, val_indices)
    
    try:
        # Load the saved model using config.model_save_path
        model = ESMCClassifier(client, DeepProteinClassifier(config.num_classes))
        state_dict = torch.load(config.model_save_path, weights_only = False)
        model.load_state_dict(state_dict['model_state_dict'])
        model = model.to(config.device)
        print(f"Loaded model from {config.model_save_path}")
    except FileNotFoundError:
        print("Saved model not found, using freshly trained model")
        model = trained_model
    
    # Get class names
    # Original class-to-index dictionary
    class_to_idx = {
        'Cytoplasm': 0,
        'Cytoplasmic Vesicle': 1,
        'Cytoskeleton': 2,
        'Endoplasmic Reticulum': 3,
        'Endosome': 4,
        'Extracellular Region': 5,
        'Golgi Apparatus': 6,
        'Lysosome': 7,
        'Microtubule': 8,
        'Mitochondrion': 9,
        'Nuclear Membrane': 10,
        'Nucleus': 11,
        'Perinuclear Region': 12,
        'Plasma Membrane': 13
    }

    # Reverse it to get index-to-class name mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Sorted list of class names by index
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]


    
    # Generate predictions on the validation set
    preds, labels = evaluate_model(model, val_dataset)
    
    # Generate and save confusion matrix
    cm_path = plot_confusion_matrix(labels, preds, class_names)

    wandb.log({"validation_confusion_matrix": wandb.Image(cm_path)})
    
    # # Print classification report
    # print("\nValidation Set Classification Report:")
    # print(classification_report(labels, preds, target_names=class_names))
    
    # Log validation metrics
    val_acc = (preds == labels).mean()
    val_f1 = f1_score(labels, preds, average='weighted')
   
    wandb.log({
        "validation_accuracy": val_acc,
        "validation_f1": val_f1
    })
    print(f"\nValidation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}")
    
    # Final checksum for reproducibility verification
    sample_checksum = hash(tuple(preds[:10].tolist()))
    print(f"Validation predictions checksum: {sample_checksum}")
    
    wandb.log({"val_pred_checksum": sample_checksum})