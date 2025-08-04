import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from kpmp_model import PatientLevelMultiTaskModel

glom_dim = 768
tile_dim = 1536
num_lesion_classes = 2
num_heads = 8
hidden_dim = 1024
dropout = 0.1

batch_size = 1
learning_rate = 1e-4
weight_decay = 1e-5
num_epochs = 100
num_workers = 4

glom_lesion_loss_weight = 1.0
patient_lesion_loss_weight = 1.0
dn_loss_weight = 2.0

train_csv = "train.csv"
dn_labels_csv = "train_dn_labels.csv"  
glom_dir = "/orange/pinaki.sarder/Davy_Jones_Locker/KPMP/Batch_1_Glom Embeddings_Gigapath" 
tile_dir = "/orange/pinaki.sarder/Davy_Jones_Locker/KPMP/Batch_1_Tile Embeddings_Gigapath"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_patient_labels(dn_csv_path):
    dn_df = pd.read_csv(dn_csv_path)
    dn_labels_dict = dict(zip(dn_df['slide_id'], dn_df['dn_label']))
    return dn_labels_dict

class KPMPPatientDataset(Dataset):
    def __init__(self, csv_path, glom_dir, tile_dir, dn_labels_dict=None, multi_label=False):
        self.df = pd.read_csv(csv_path)
        self.glom_dir = glom_dir
        self.tile_dir = tile_dir
        self.dn_labels_dict = dn_labels_dict
        self.multi_label = multi_label
        
        # Group by slide_id (unqiue slide_id means a particular patient)
        self.patient_groups = self.df.groupby('slide_id')
        self.slide_ids = list(self.patient_groups.groups.keys())
        print(f"Dataset: {len(self.slide_ids)} patients, {len(self.df)} glomeruli")
        
    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        patient_gloms = self.patient_groups.get_group(slide_id)
        
        slide_name_no_svs = slide_id.replace('.svs', '')
        
        # Load tile embeddings
        tile_path = os.path.join(
            self.tile_dir, 
            slide_name_no_svs, 
            f"{slide_name_no_svs}_tile_encoder_outputs.pt"
        )
            
        tile_data = torch.load(tile_path, map_location='cpu')
        tile_embs = tile_data['tile_embeds']
        coords = tile_data['coords'].float()
        
        # Normalize coordinates
        if coords.shape[0] > 0:
            coords_min = coords.min(dim=0)[0]
            coords_max = coords.max(dim=0)[0]
            coords_range = coords_max - coords_min + 1e-6
            coords = (coords - coords_min) / coords_range
        
        # Load glom embeddings and labels
        glom_embs = []
        lesion_labels = []
        glom_names = []
        
        for _, row in patient_gloms.iterrows():
            glom_name = row['name'].replace('.png', '')
            if self.multi_label:
                lesion_label = eval(row['ground truth']) if isinstance(row['ground truth'], str) else row['ground truth']
                lesion_label = torch.tensor(lesion_label, dtype=torch.float32)
            else:
                lesion_label = int(row['ground truth'])
            
            # Load glom embedding
            glom_path = os.path.join(self.glom_dir, f"{glom_name}.pt")
                
            glom_data = torch.load(glom_path, map_location='cpu')
            glom_emb = glom_data['last_layer_embed']
            
            glom_embs.append(glom_emb)
            lesion_labels.append(lesion_label)
            glom_names.append(glom_name)
        
        # Stack embeddings
        glom_embs = torch.stack(glom_embs, dim=0)
        
        if self.multi_label:
            lesion_labels = torch.stack(lesion_labels, dim=0)
            patient_lesion_probs = lesion_labels.mean(dim=0)
        else:
            lesion_labels = torch.tensor(lesion_labels, dtype=torch.long)
            lesion_counts = torch.bincount(lesion_labels, minlength=num_lesion_classes)
            patient_lesion_probs = lesion_counts.float() / len(lesion_labels)
        
        return_dict = {
            'glom_embs': glom_embs,
            'tile_embs': tile_embs,
            'coords': coords,
            'glom_lesion_labels': lesion_labels,
            'patient_lesion_probs': patient_lesion_probs,
            'slide_id': slide_id,
            'glom_names': glom_names,
            'num_gloms': glom_embs.shape[0],
            'num_tiles': tile_embs.shape[0]
        }
        
        # Add DN label if available
        if self.dn_labels_dict is not None:
            patient_dn_label = self.dn_labels_dict.get(slide_id, 0)
            return_dict['patient_dn_label'] = torch.tensor(patient_dn_label, dtype=torch.float32)
        
        return return_dict

def collate_fn(batch):
    max_gloms = max(sample['num_gloms'] for sample in batch)
    max_tiles = max(sample['num_tiles'] for sample in batch)
    
    b = len(batch)
    glom_embs = torch.zeros(b, max_gloms, glom_dim)
    tile_embs = torch.zeros(b, max_tiles, tile_dim)
    coords = torch.zeros(b, max_tiles, 2)
    patient_lesion_probs = torch.zeros(b, num_lesion_classes)
    
    # Handle single vs multi-label
    first_sample = batch[0]['glom_lesion_labels']
    if first_sample.dim() == 1:
        glom_lesion_labels = torch.zeros(b, max_gloms, dtype=torch.long)
    else:
        glom_lesion_labels = torch.zeros(b, max_gloms, num_lesion_classes, dtype=torch.float32)
    
    # Masks for padding
    glom_masks = torch.zeros(b, max_gloms, dtype=torch.bool)
    tile_masks = torch.zeros(b, max_tiles, dtype=torch.bool)
    
    slide_ids = []
    glom_names_batch = []
    
    # Check for DN labels
    has_dn_labels = 'patient_dn_label' in batch[0]
    if has_dn_labels:
        patient_dn_labels = torch.zeros(b, dtype=torch.float32)
    
    for i, sample in enumerate(batch):
        n_gloms = sample['num_gloms']
        n_tiles = sample['num_tiles']
        
        # Handle dimension mismatch
        sample_glom_embs = sample['glom_embs']
        if sample_glom_embs.dim() == 3 and sample_glom_embs.shape[1] == 1:
            sample_glom_embs = sample_glom_embs.squeeze(1)
        
        # Fill data
        glom_embs[i, :n_gloms] = sample_glom_embs
        tile_embs[i, :n_tiles] = sample['tile_embs']
        coords[i, :n_tiles] = sample['coords']
        glom_lesion_labels[i, :n_gloms] = sample['glom_lesion_labels']
        patient_lesion_probs[i] = sample['patient_lesion_probs']
        
        if has_dn_labels:
            patient_dn_labels[i] = sample['patient_dn_label']
        
        # Set masks
        glom_masks[i, n_gloms:] = True
        tile_masks[i, n_tiles:] = True
        
        slide_ids.append(sample['slide_id'])
        glom_names_batch.append(sample['glom_names'])
    
    return_dict = {
        'glom_embs': glom_embs,
        'tile_embs': tile_embs,
        'coords': coords,
        'glom_lesion_labels': glom_lesion_labels,
        'patient_lesion_probs': patient_lesion_probs,
        'glom_mask': glom_masks,
        'tile_mask': tile_masks,
        'slide_ids': slide_ids,
        'glom_names': glom_names_batch
    }
    
    if has_dn_labels:
        return_dict['patient_dn_labels'] = patient_dn_labels
    
    return return_dict

def train_epoch(model, dataloader, optimizer, device, multi_label=False):
    model.train()
    
    total_glom_lesion_loss = 0
    total_patient_lesion_loss = 0
    total_dn_loss = 0
    total_loss = 0
    glom_lesion_correct = 0
    glom_lesion_total = 0
    dn_correct = 0
    dn_total = 0
    patient_predictions_all_classes = []  
    patient_true_proportions_all_classes = [] 
    
    # Loss functions
    if multi_label:
        lesion_criterion = nn.BCELoss()
    else:
        lesion_criterion = nn.CrossEntropyLoss()
    patient_lesion_criterion = nn.MSELoss()
    dn_criterion = nn.BCELoss()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        glom_embs = batch['glom_embs'].to(device)
        tile_embs = batch['tile_embs'].to(device)
        coords = batch['coords'].to(device)
        glom_mask = batch['glom_mask'].to(device)
        tile_mask = batch['tile_mask'].to(device)
        glom_lesion_labels = batch['glom_lesion_labels'].to(device)
        patient_lesion_probs = batch['patient_lesion_probs'].to(device)
        
        # Forward pass
        outputs = model(glom_embs, tile_embs, coords, glom_mask, tile_mask)
        
        # Glom lesion loss
        glom_lesion_logits = outputs['glom_lesion_logits']
        glom_lesion_loss = 0
        valid_gloms = ~glom_mask
        
        for b in range(glom_embs.shape[0]):
            valid_mask = valid_gloms[b]
            if valid_mask.sum() > 0:
                valid_lesion_logits = glom_lesion_logits[b][valid_mask]
                valid_lesion_labels = glom_lesion_labels[b][valid_mask]
                
                if multi_label:
                    glom_lesion_loss += lesion_criterion(valid_lesion_logits, valid_lesion_labels)
                    preds = (valid_lesion_logits > 0.5).float()
                    glom_lesion_correct += (preds == valid_lesion_labels).all(dim=1).sum().item()
                else:
                    glom_lesion_loss += lesion_criterion(valid_lesion_logits, valid_lesion_labels)
                    preds = valid_lesion_logits.argmax(dim=1)
                    glom_lesion_correct += (preds == valid_lesion_labels).sum().item()
                
                glom_lesion_total += valid_lesion_labels.size(0)
        
        glom_lesion_loss = glom_lesion_loss / glom_embs.shape[0]
        
        # Patient lesion loss
        patient_lesion_probs_pred = outputs['patient_lesion_logits']
        patient_lesion_loss = patient_lesion_criterion(patient_lesion_probs_pred, patient_lesion_probs)
        
        # Save All class predictions and true proportions for correlation
        patient_predictions_all_classes.append(patient_lesion_probs_pred.detach().cpu().numpy())
        patient_true_proportions_all_classes.append(patient_lesion_probs.detach().cpu().numpy())
        
        # DN loss
        dn_loss = torch.tensor(0.0, device=device)
        if 'patient_dn_labels' in batch:
            patient_dn_labels = batch['patient_dn_labels'].to(device)
            dn_predictions = outputs['dn_probability'].squeeze(-1)
            dn_loss = dn_criterion(dn_predictions, patient_dn_labels)
            
            dn_preds = (dn_predictions > 0.5).float()
            dn_correct += (dn_preds == patient_dn_labels).sum().item()
            dn_total += patient_dn_labels.size(0)
        
        # Combined loss
        total_batch_loss = (glom_lesion_loss_weight * glom_lesion_loss + 
                          patient_lesion_loss_weight * patient_lesion_loss +
                          dn_loss_weight * dn_loss)
        
        # Backward pass
        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_glom_lesion_loss += glom_lesion_loss.item()
        total_patient_lesion_loss += patient_lesion_loss.item()
        total_dn_loss += dn_loss.item() if isinstance(dn_loss, torch.Tensor) else 0
        total_loss += total_batch_loss.item()
        
        # Progress bar
        pbar.set_postfix({
            'glom_loss': f'{glom_lesion_loss.item():.4f}',
            'patient_loss': f'{patient_lesion_loss.item():.4f}',
            'dn_loss': f'{dn_loss.item() if isinstance(dn_loss, torch.Tensor) else 0:.4f}',
            'glom_acc': f'{glom_lesion_correct/max(glom_lesion_total, 1):.4f}'
        })
    
    # Calculate correlation for ALL classes combined
    patient_correlation = 0.0
    if len(patient_predictions_all_classes) > 1:
        pred_flat = np.concatenate(patient_predictions_all_classes, axis=0).flatten()
        true_flat = np.concatenate(patient_true_proportions_all_classes, axis=0).flatten()
        
        if len(pred_flat) > 1 and len(true_flat) > 1:
            patient_correlation = np.corrcoef(true_flat, pred_flat)[0, 1]
            if np.isnan(patient_correlation):
                patient_correlation = 0.0
    
    return {
        'glom_lesion_loss': total_glom_lesion_loss / len(dataloader),
        'patient_lesion_loss': total_patient_lesion_loss / len(dataloader),
        'dn_loss': total_dn_loss / len(dataloader),
        'total_loss': total_loss / len(dataloader),
        'glom_lesion_accuracy': glom_lesion_correct / max(glom_lesion_total, 1),
        'dn_accuracy': dn_correct / max(dn_total, 1),
        'patient_correlation': patient_correlation,
    }

def main():
    
    multi_label = False  
    
    try:
        dn_labels_dict = load_patient_labels(dn_labels_csv)
    except FileNotFoundError:
        return None, None
    
    # Create dataset
    train_dataset = KPMPPatientDataset(
        train_csv, glom_dir, tile_dir,
        dn_labels_dict=dn_labels_dict,
        multi_label=multi_label
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    model = PatientLevelMultiTaskModel(
        glom_dim=glom_dim,
        tile_dim=tile_dim,
        num_lesion_classes=num_lesion_classes,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout,
        multi_label=multi_label
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01
    )
    
    # Training loop
    train_logs = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_metrics = train_epoch(model, train_dataloader, optimizer, device, multi_label)
        
        print(f"Losses: Glom={train_metrics['glom_lesion_loss']:.4f}, "
              f"Patient={train_metrics['patient_lesion_loss']:.4f}, "
              f"DN={train_metrics['dn_loss']:.4f}")
        print(f"Accuracy: Glom={train_metrics['glom_lesion_accuracy']:.4f}, "
              f"DN={train_metrics['dn_accuracy']:.4f}")
        print(f"Patient Correlation (All Classes): {train_metrics['patient_correlation']:.4f}")
        
        if hasattr(model, 'lesion_coefficients'):
            with torch.no_grad():
                coeffs = torch.softmax(model.lesion_coefficients, dim=0).cpu().numpy()
                print(f"Lesion Weights: Class 0={coeffs[0]:.3f}, Class 1={coeffs[1]:.3f}")
        
        # Save metrics
        train_metrics['epoch'] = epoch + 1
        train_metrics['lr'] = optimizer.param_groups[0]['lr']
        train_logs.append(train_metrics)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_logs': train_logs,
            }, checkpoint_path)
            
           #Progress summary
            print(f"Glom Accuracy: {train_metrics['glom_lesion_accuracy']:.1%}")
            print(f"Patient Correlation: {train_metrics['patient_correlation']:.3f}")
    
    # Save final model
    final_model_path = 'final_patient_level_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_logs': train_logs
    }, final_model_path)
    
    # Save lesion encodings
    with torch.no_grad():
        lesion_encodings = model.lesion_encoder.lesion_embeddings.cpu()
        torch.save(lesion_encodings, 'lesion_encodings.pt')
    
    # Save training logs
    train_log_df = pd.DataFrame(train_logs)
    train_log_df.to_csv("patient_level_training_logs.csv", index=False)
    
    return model, train_logs

if __name__ == "__main__":
    model, train_logs = main()