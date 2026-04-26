import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from image_rnn_model import ImageAutoregressiveRNN


# COUNTRIES: ['CA', 'US', 'ES', 'JP', 'UK', 'MX', 'IT', 'BR', 'DE', 'FR']
COUNTRY2IDX = {'CA': 0, 'US': 1, 'ES': 2, 'JP': 3, 'UK': 4, 'MX': 5, 'IT': 6, 'BR': 7, 'DE': 8, 'FR': 9}
OS2IDX = {'iOS': 0, 'Android': 1}

class CreativeImageDataset(Dataset):
    """
    Highly Optimized Dataset:
    Pre-computes all CTR sequences in __init__ to avoid Pandas lookups in __getitem__.
    """
    def __init__(self, seq_len=30):
        print("Loading and pre-processing dataset (this takes a few seconds but makes training lightning fast)...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../../../../frontend/public/data")
        
        self.creatives = pd.read_csv(os.path.join(data_dir, "creatives.csv"))
        df_stats = pd.read_csv(os.path.join(data_dir, "creative_daily_country_os_stats.csv"))
        
        # 1. Aggregate CTR
        agg_stats = df_stats.groupby(['creative_id', 'country', 'os', 'days_since_launch'])[['clicks', 'impressions']].sum().reset_index()
        agg_stats['CTR'] = agg_stats['clicks'] / (agg_stats['impressions'] + 1e-5)
        
        self.base_path = data_dir
        self.seq_len = seq_len
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 2. Create a fast lookup dictionary for asset files
        creative_asset_map = dict(zip(self.creatives['creative_id'], self.creatives['asset_file']))
        
        # 3. Pre-process everything into a fast Python list
        self.samples = []
        
        # Group the stats by cohort so we only loop over them once
        grouped = agg_stats.groupby(['creative_id', 'country', 'os'])
        
        for (cid, country_str, os_str), group in grouped:
            if cid not in creative_asset_map:
                continue
                
            asset_file = creative_asset_map[cid]
            img_path = os.path.join(self.base_path, str(asset_file))
            
            if not os.path.exists(img_path):
                continue
                
            # Sort chronologically and extract the CTR array once
            group = group.sort_values('days_since_launch')
            ctr_array = group['CTR'].values * 100.0 # Scale here to save time later
            
            # Store purely native Python/Numpy objects in the sample list
            self.samples.append({
                'img_path': img_path,
                'country_idx': COUNTRY2IDX.get(country_str, 0),
                'os_idx': OS2IDX.get(os_str, 0),
                'ctr_array': ctr_array
            })
            
        print(f"Pre-processing complete. Found {len(self.samples)} valid sequence cohorts.")
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # 1. Grab pre-computed dictionary (O(1) lookup speed)
        sample = self.samples[idx]
        
        # 2. Load the Image (this is now the only I/O operation here)
        image = Image.open(sample['img_path']).convert('RGB')
        image = self.transform(image)
        
        # 3. Build Target & Mask sequences
        ctr_array = sample['ctr_array']
        target = torch.zeros(self.seq_len, 1)
        mask = torch.zeros(self.seq_len, 1)
        
        actual_len = min(len(ctr_array), self.seq_len)
        if actual_len > 0:
            target[:actual_len, 0] = torch.tensor(ctr_array[:actual_len], dtype=torch.float32)
            mask[:actual_len, 0] = 1.0
                
        country_idx = torch.tensor(sample['country_idx'], dtype=torch.long)
        os_idx = torch.tensor(sample['os_idx'], dtype=torch.long)
            
        return image, country_idx, os_idx, target, mask

def train():
    print("Initializing Image RNN model data structures...")
    dataset = CreativeImageDataset()
    
    if len(dataset) == 0:
        print("No valid data found. Please ensure assets and stats are generated.")
        return
        
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImageAutoregressiveRNN(
        latent_dim=128, 
        hidden_dim=64,
        num_countries=len(COUNTRY2IDX),
        num_os=len(OS2IDX)
    ).to(device)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    print(f"Training Image RNN using {len(dataset)} unique sequence cohorts...")
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        current_tf_ratio = max(0.1, 1.0 - (epoch / epochs))
        
        for images, countries, os_types, targets, masks in loader:
            
            images = images.to(device)
            countries = countries.to(device)
            os_types = os_types.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            preds = model(
                images, 
                country_idx=countries, 
                os_idx=os_types, 
                targets=targets, 
                teacher_forcing_ratio=current_tf_ratio
            )
            
            # --- MASKED LOSS CALCULATION ---
            masked_preds = preds * masks
            masked_targets = targets * masks
            
            sum_squared_error = F.mse_loss(masked_preds, masked_targets, reduction='sum')
            loss = sum_squared_error / (masks.sum() + 1e-8)
            # -------------------------------
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Masked MSE Loss: {total_loss/len(loader):.6f}")
        
    model_path = os.path.join(os.path.dirname(__file__), "image_rnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel training complete! Successfully saved weights to: {model_path}")


if __name__ == '__main__':
    train()