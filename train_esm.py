## changed the attention mechanism to the attention layer claculated previously
## changed to simpler attention as fromt he deeploc paper that ahs only query
import os
import random
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
# from peft import get_peft_model, LoraConfig, TaskType
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
from scipy.fftpack import dct


# Set all random seeds for reproducibility
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


set_seed(42)


# Initialize WandB
# os.environ['WANDB_DISABLED'] = 'true'  # Add this line


os.environ["TOKENIZERS_PARALLELISM"] = "false"


wandb.init(project="all_mammel_tm", name="newds")


# Configuration
class Config:
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model_name_or_path = "esmc_300m"
   resume_training = False
   training = True
   # UPDATED: Path to your ESM data CSV
   label_csv = '/work/hdd/bdja/bpokhrel/esm_new/esmdata_20251103_001643.csv'
   batch_size = 35
   num_epochs = 250
   patience = 15
   lr_esmc = 1e-5
   lr_classifier = 3e-5
   weight_decay = 1e-3
   max_len = 1000
   test_mode = False #False
   only_validate = False
   entropy_weight = 0.25
   # save_cm = '/Users/bivekpokhrel/Desktop/bio/deeploc_dataset/deeploc_py/noloraconfusion_matrix_dct2.png'
   model_save_path = '/work/hdd/bdja/bpokhrel/esm_new/newds_best.pth'
   last_model_save_path = '/work/hdd/bdja/bpokhrel/esm_new/newds_last.pth'  # Add this line
   output_dir = '/work/hdd/bdja/bpokhrel/esm_new/newds'
   os.makedirs(output_dir, exist_ok=True)

   # UPDATED: Based on your localization classes from the TM dataset
   num_classes = 6  # Must match your actual class count
   classes = ['Endoplasmic Reticulum',
    'Golgi Apparatus',
    'Lysosome',
    'Mitochondrion',
    'Peroxisome',
    'Plasma Membrane']


config = Config()

class ProteinDataset(Dataset):
    def __init__(self, label_csv, max_len=2000, test_mode=False):
        self.max_len = max_len
        self.test_mode = test_mode
        
        # Load your ESM data CSV
        df = pd.read_csv(label_csv)
        
        # UPDATED: Map your CSV columns to expected format using your exact headers
        self.label_mapping = {loc: idx for idx, loc in enumerate(config.classes)}
        
        # Filter and prepare data
        df = df[df['loc_normalized'].isin(config.classes)].copy()
        df['Sequence'] = df['sequence']  # Use 'sequence' column from your CSV
        df['Sequence'] = df['Sequence'].apply(self.truncate_sequence)
        df['length'] = df['Sequence'].apply(len)
        
        # Map string labels to numerical labels
        df['label'] = df['loc_normalized'].map(self.label_mapping)
        
        # Remove any rows with NaN labels (if loc_normalized not in our classes)
        df = df.dropna(subset=['label']).copy()
        df['label'] = df['label'].astype(int)
        
        if test_mode:
            df = df.head(1000).copy()
            
        self.sequences = df['Sequence'].tolist()
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        self.lengths = df['length'].tolist()
        self.ids = df['accession'].tolist()  # Use 'accession' column
        self.cluster_ids = df['cluster_id'].tolist()  # Use 'cluster_id' column
        
        # Calculate class weights
        class_counts = torch.bincount(self.labels)
        self.class_weights = 1 / torch.sqrt(class_counts.float() + 10)
        self.class_weights = self.class_weights / self.class_weights.max()
        self.df = df.copy()

        print(f"Dataset loaded: {len(self.sequences)} sequences")
        print(f"Class distribution: {dict(zip(config.classes, class_counts.tolist()))}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.labels[idx],
            self.lengths[idx],
            self.ids[idx],
            self.cluster_ids[idx]
        )
    
    def truncate_sequence(self, sequence):
        if len(sequence) <= self.max_len:
            return sequence
        front_len = self.max_len // 2
        back_len = self.max_len - front_len
        return sequence[:front_len] + sequence[-back_len:]

def collate_fn(batch):
    sequences, labels, lengths, ids, cluster_ids = zip(*batch)
    return {
        "sequences": sequences,
        "labels": torch.stack(labels),
        "lengths": torch.tensor(lengths),
        "ids": ids,
        "cluster_ids": cluster_ids
    }

def create_dataloaders():
    dataset = ProteinDataset(config.label_csv, max_len=config.max_len, test_mode=config.test_mode)
    
    # Create a DataFrame for easier manipulation
    data_df = pd.DataFrame({
        'index': range(len(dataset)),
        'label': [item[1].item() for item in dataset],
        'cluster_id': [item[4] for item in dataset]
    })
    
    # Get unique cluster_id + label pairs
    cluster_label_pairs = data_df[['cluster_id', 'label']].drop_duplicates()
    
    # Stratified split on the pairs (stratify by label to maintain class balance)
    train_pairs, val_pairs = train_test_split(
        cluster_label_pairs,
        test_size=0.3,
        random_state=42,
        stratify=cluster_label_pairs['label']
    )
    
    # Get indices of all proteins belonging to these pairs
    train_indices = data_df.merge(train_pairs, on=['cluster_id', 'label'])['index'].tolist()
    val_indices = data_df.merge(val_pairs, on=['cluster_id', 'label'])['index'].tolist()
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"Training set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4 if not config.test_mode else 0
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4 if not config.test_mode else 0
    )
    
    return train_loader, val_loader

# Rest of your model classes remain the same...
class LocalizationAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
       
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
       
        for module in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(module.weight, gain=1/(num_heads**0.25))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        x = x.to(next(self.parameters()).device)
        if mask is not None:
            mask = mask.to(x.device)
        
        B, L, _ = x.shape
        x = self.layer_norm(x)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.25)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, -1e4)
        
        if mask is not None:
            seq_lengths = mask.sum(dim=1)
            
            for i in range(B):
                actual_len = int(seq_lengths[i].item())
                
                if actual_len >= 20:
                    scores[i, :, :, :20] += 1.0
                    scores[i, :, :, actual_len-20:actual_len] += 0.8
                else:
                    scores[i, :, :, :actual_len] += 1.0 + 0.8
        else:
            if L >= 20:
                scores[:, :, :, :20] += 1.0
                scores[:, :, :, -20:] += 0.8
        
        attn_weights = F.softmax(scores, dim=-1)
        entropy_loss = self._entropy_regularization(attn_weights)
        context = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, L, -1)
        
        if mask is not None:
            seq_lengths = mask.sum(dim=1)
            
            n_term_list = []
            for i in range(B):
                actual_len = int(seq_lengths[i].item())
                n_term_size = min(20, actual_len)
                n_term_slice = context[i, :n_term_size]
                n_term_list.append(n_term_slice.mean(dim=0))
            n_term = torch.stack(n_term_list)
            
            c_term_list = []
            for i in range(B):
                actual_len = int(seq_lengths[i].item())
                c_term_size = min(20, actual_len)
                start_idx = max(0, actual_len - c_term_size)
                c_term_slice = context[i, start_idx:actual_len]
                c_term_list.append(c_term_slice.mean(dim=0))
            c_term = torch.stack(c_term_list)
        else:
            n_term_size = min(20, L)
            c_term_size = min(20, L)
            n_term = context[:, :n_term_size].mean(dim=1)
            c_term = context[:, -c_term_size:].mean(dim=1)
        
        global_pool = context.mean(dim=1)
        pooled = torch.cat([n_term, c_term, global_pool], dim=1)
        
        return pooled, attn_weights.mean(dim=1), entropy_loss

    def _entropy_regularization(self, attn_weights):
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
        return entropy.mean()

class ProteinClassifier(nn.Module):
   def __init__(self, num_classes):
       super().__init__()
       self.attention = LocalizationAttention(embed_dim=960)
       self.classifier = nn.Sequential(
           nn.Linear(960 * 3, 512),
           nn.ReLU(),
           nn.Dropout(0.4),
           nn.Linear(512, num_classes))

   def forward(self, x, mask=None):
       x = x.to(config.device)
       if mask is not None:
           mask = mask.to(config.device)
          
       context, attn_weights, entropy_loss = self.attention(x, mask)
       logits = self.classifier(context)
       return {
           'logits': logits,
           'attention': attn_weights,
           'entropy_loss': entropy_loss
       }

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
               LogitsConfig(return_embeddings=False, return_hidden_states=True)
           )
           hidden_s = outputs.hidden_states[1:]
           hidden_s = hidden_s.squeeze(1)
           embed = torch.mean(hidden_s, dim=0)
           embeddings.append(embed)
      
       lengths = batch["lengths"].tolist()
       max_len = max(lengths)
       batch_size = len(embeddings)
      
       padded_embeddings = torch.zeros((batch_size, max_len, 960), device=config.device)
       attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=config.device)
      
       for i, (embed, seq_len) in enumerate(zip(embeddings, lengths)):
           padded_embeddings[i, :seq_len] = embed[:seq_len]
           attention_mask[i, :seq_len] = 1
      
       classifier_output = self.classifier(padded_embeddings, attention_mask)
      
       return {
           'logits': classifier_output['logits'],
           'attention': classifier_output['attention'],
           'entropy_loss': classifier_output['entropy_loss'],
       }

# Rest of your training and validation code remains exactly the same...
def train_model(resume_from_checkpoint=None, train_loader=None, val_loader=None, model=None):
   """Complete training loop with device handling and early stopping"""
  
   class_weights = train_loader.dataset.dataset.class_weights.to(config.device)
   print("\nUsing class weights:", class_weights)
  
   start_epoch = 0
   best_val_loss = float('inf')
   best_val_f1 = 0.0
   epochs_no_improve = 0
  
   optimizer = optim.AdamW(
       [
           {'params': model.esmc.parameters(), 'lr': config.lr_esmc},
           {'params': model.classifier.parameters(), 'lr': config.lr_classifier}
       ],
       weight_decay=config.weight_decay,
       betas=(0.9, 0.98)
   )
  
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(
   optimizer,
   mode='max',
   factor=0.5,
   patience=8,
   verbose=True,
   threshold=0.0001,
   min_lr=1e-7
   )
  
   criterion = nn.CrossEntropyLoss(weight=class_weights)
  
   if resume_from_checkpoint:
       checkpoint = torch.load(resume_from_checkpoint)
       model.load_state_dict(checkpoint['model_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
       start_epoch = checkpoint['epoch'] + 1
       best_val_loss = checkpoint['best_val_loss']
       best_val_f1 = checkpoint['best_val_f1']
       epochs_no_improve = checkpoint['epochs_no_improve']
       print(f"Resuming from epoch {start_epoch} with best val F1: {best_val_f1:.4f}")
  
   for epoch in range(start_epoch, config.num_epochs):
       if epochs_no_improve >= config.patience:
           print(f"Early stopping after {epochs_no_improve} epochs without improvement")
           break
          
       model.train()
       train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
      
       for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
           labels = batch["labels"].to(config.device)
           lengths = batch["lengths"].to(config.device)
          
           optimizer.zero_grad()
          
           outputs = model(batch)
           logits = outputs['logits']
           entropy_loss = outputs['entropy_loss']
          
           loss = criterion(logits, labels) + config.entropy_weight * entropy_loss
          
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
          
           with torch.no_grad():
               preds = logits.argmax(dim=1)
               acc = (preds == labels).float().mean()
               f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
              
           train_loss += loss.item()
           train_acc += acc.item()
           train_f1 += f1
          
           if batch_idx % 10 == 0:
               wandb.log({
                   "batch/train_loss": loss.item(),
                   "batch/train_acc": acc.item(),
                   "batch/train_f1": f1,
                   "batch/lr": scheduler.get_last_lr()[0]
               })
      
       model.eval()
       val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
       all_preds = []
       all_labels = []
      
       with torch.no_grad():
           for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
               labels = batch["labels"].to(config.device)
               lengths = batch["lengths"].to(config.device)
              
               outputs = model(batch)
               logits = outputs['logits']
               entropy_loss = outputs['entropy_loss']
              
               loss = criterion(logits, labels) + config.entropy_weight * entropy_loss
              
               preds = logits.argmax(dim=1)
               acc = (preds == labels).float().mean()
               f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
              
               val_loss += loss.item()
               val_acc += acc.item()
               val_f1 += f1
               all_preds.extend(preds.cpu().numpy())
               all_labels.extend(labels.cpu().numpy())
      
       train_loss /= len(train_loader)
       train_acc /= len(train_loader)
       train_f1 /= len(train_loader)
       val_loss /= len(val_loader)
       val_acc /= len(val_loader)
       val_f1 /= len(val_loader)

       scheduler.step(val_f1)
      
       if val_f1 > best_val_f1:
           best_val_f1 = val_f1
           best_val_loss = val_loss
           epochs_no_improve = 0
           torch.save({
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(),
               'epoch': epoch,
               'best_val_loss': val_loss,
               'best_val_f1': val_f1,
               'epochs_no_improve': epochs_no_improve
           }, config.model_save_path)
           print(f"Saved new best model to {config.model_save_path}")
       else:
           epochs_no_improve += 1
      
       torch.save({
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler_state_dict': scheduler.state_dict(),
           'epoch': epoch,
           'best_val_loss': best_val_loss,
           'best_val_f1': best_val_f1,
           'epochs_no_improve': epochs_no_improve
       }, config.last_model_save_path)
      
       wandb.log({
           "epoch": epoch,
           "train/loss": train_loss,
           "train/acc": train_acc,
           "train/f1": train_f1,
           "val/loss": val_loss,
           "val/acc": val_acc,
           "val/f1": val_f1,
           "best_val/f1": best_val_f1
       })
      
       print(f"\nEpoch {epoch+1}/{config.num_epochs}:")
       print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
       print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
       print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
       print(f"Best Val F1: {best_val_f1:.4f} | Epochs w/o improvement: {epochs_no_improve}")
  
   print("\nTraining complete!")
   print(f"Best Validation F1: {best_val_f1:.4f}")
  
   return model

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    # Initialize dataloaders and model
    train_loader, val_loader = create_dataloaders()
    client = ESMC.from_pretrained(config.model_name_or_path).to(config.device)
    classifier = ProteinClassifier(config.num_classes).to(config.device)
    model = ESMCClassifier(client, classifier).to(config.device)

    # Define loss function with class weights
    class_weights = train_loader.dataset.dataset.class_weights.to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if config.resume_training and os.path.exists(config.last_model_save_path):
        checkpoint = torch.load(config.last_model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    if config.training:
        model = train_model(
            resume_from_checkpoint=config.last_model_save_path if config.resume_training else None,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model
        )
    elif config.only_validate:
        # Load best model checkpoint
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_ids = []
        all_attention = []
        all_lengths = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                labels = batch["labels"].to(config.device)
                ids = batch["ids"]
                lengths = batch["lengths"].cpu().numpy()
                
                outputs = model(batch)
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)
                attention = outputs['attention']
                
                loss = criterion(logits, labels) + config.entropy_weight * outputs['entropy_loss']
                
                preds = logits.argmax(dim=1)
                
                val_loss += loss.item()
                val_acc += (preds == labels).float().mean().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_ids.extend(ids)
                all_lengths.extend(lengths)
                
                for i in range(len(ids)):
                    sample_attention = attention[i].cpu().numpy()
                    all_attention.append(sample_attention[:lengths[i]].tolist())

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')

        print("\nValidation Results:")
        print(f"Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Weighted F1: {val_f1:.4f} | Macro F1: {val_f1_macro:.4f}")
        print(classification_report(all_labels, all_preds, target_names=config.classes))

        val_results = pd.DataFrame({
            'protein_id': all_ids,
            'sequence_length': all_lengths,
            'true_label': all_labels,
            'predicted_label': all_preds,
            'true_class': [config.classes[x] for x in all_labels],
            'predicted_class': [config.classes[x] for x in all_preds],
            **{f'prob_class_{i}': np.array(all_probs)[:, i] for i in range(config.num_classes)},
            'attention_weights': all_attention
        })
        
        output_path = os.path.join(config.output_dir, 'validation_results_with_attention.csv')
        val_results.to_csv(output_path, index=False)
        print(f"\nSaved detailed validation results with attention to: {output_path}")
        
        attention_df = pd.DataFrame({
            'protein_id': all_ids,
            'attention_weights': all_attention,
            'sequence_length': all_lengths
        })
        attention_path = os.path.join(config.output_dir, 'attention_weights.json')
        attention_df.to_json(attention_path, orient='records', indent=2)
        print(f"Saved attention weights to: {attention_path}")