import os
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Import your trained model architecture
from image_rnn_model import ImageAutoregressiveRNN

# --- 1. Global Mappings ---
COUNTRY2IDX = {'CA': 0, 'US': 1, 'ES': 2, 'JP': 3, 'UK': 4, 'MX': 5, 'IT': 6, 'BR': 7, 'DE': 8, 'FR': 9}
OS2IDX = {'iOS': 0, 'Android': 1}

# --- 2. Simulation Function ---
_image_rnn_model = None
def get_image_ctr_timeseries(image_path: str, country: str, os_type: str, seq_len: int = 30) -> list[float]:
    global _image_rnn_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if _image_rnn_model is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_rnn_model.pth")
        _image_rnn_model = ImageAutoregressiveRNN(
            latent_dim=128, 
            hidden_dim=64, 
            num_countries=len(COUNTRY2IDX), 
            num_os=len(OS2IDX)
        ).to(device)
        if os.path.exists(model_path):
            _image_rnn_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        _image_rnn_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return [0.0] * seq_len
        
    image_tensor = transform(image).unsqueeze(0).to(device)
    preds = simulate_demographic_creative(_image_rnn_model, image_tensor, country, os_type, seq_len=seq_len, device=device)
    return preds.tolist()

def simulate_demographic_creative(model, image_tensor, country_str, os_str, seq_len=50, device='cpu'):
    model.eval()
    
    # Safely get indices
    country_idx = torch.tensor([COUNTRY2IDX.get(country_str, 0)], dtype=torch.long).to(device)
    os_idx = torch.tensor([OS2IDX.get(os_str, 0)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Passing targets=None triggers the pure autoregressive generation loop
        preds = model(image_tensor, country_idx, os_idx, targets=None, seq_len=seq_len)
        
    # Squeeze out batch and feature dimensions, scale back down from percentage
    preds_np = preds.squeeze().cpu().numpy() / 100.0
    return preds_np

# --- 3. Main Evaluation Logic ---
def evaluate():
    print("Loading datasets and model...")
    
    # Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "../../../../frontend/public/data"))
    model_path = os.path.join(base_dir, "image_rnn_model.pth")
    
    # Load Data
    creatives_df = pd.read_csv(os.path.join(data_dir, "creatives.csv"))
    df_stats = pd.read_csv(os.path.join(data_dir, "creative_daily_country_os_stats.csv"))
    
    # Aggregate Ground Truth CTR
    agg_stats = df_stats.groupby(['creative_id', 'country', 'os', 'days_since_launch'])[['clicks', 'impressions']].sum().reset_index()
    agg_stats['CTR'] = agg_stats['clicks'] / (agg_stats['impressions'] + 1e-5)
    
    # Find a creative with diverse demographic data
    cohorts = agg_stats[['creative_id', 'country', 'os']].drop_duplicates()
    valid_cids = []
    
    for cid, group in cohorts.groupby('creative_id'):
        if group['country'].nunique() >= 2 and group['os'].nunique() >= 2:
            # Verify image exists
            asset_rows = creatives_df[creatives_df['creative_id'] == cid]
            if not asset_rows.empty:
                asset_file = asset_rows.iloc[0]['asset_file']
                if os.path.exists(os.path.join(data_dir, str(asset_file))):
                    valid_cids.append(cid)
                    
    if not valid_cids:
        print("Could not find a creative with 2+ countries and 2+ OSs. Please check your data.")
        return
        
    # Sample 1 random creative and 2 random demographics from its history
    sampled_cid = random.choice(valid_cids)
    cid_cohorts = cohorts[cohorts['creative_id'] == sampled_cid]
    
    sampled_countries = random.sample(list(cid_cohorts['country'].unique()), 2)
    sampled_oss = random.sample(list(cid_cohorts['os'].unique()), 2)
    
    print(f"Sampled Creative {sampled_cid}. Demographics: {sampled_countries} x {sampled_oss}")
    
    # Load Image
    asset_file = creatives_df[creatives_df['creative_id'] == sampled_cid].iloc[0]['asset_file']
    img_path = os.path.join(data_dir, str(asset_file))
    image = Image.open(img_path).convert('RGB')
    
    # Setup Device & Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageAutoregressiveRNN(
        latent_dim=128, 
        hidden_dim=64, 
        num_countries=len(COUNTRY2IDX), 
        num_os=len(OS2IDX)
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Warning: {model_path} not found. Running with untrained weights.")
        
    # Transform Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # --- 4. Plotting ---
    # Create a layout: Left half is the image, right half is a 2x2 grid of plots
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
    
    # Display Image
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_img.set_title(f"Ad Creative Image\nID: {sampled_cid}", fontsize=14, fontweight='bold')
    
    # Define the 4 combinations
    combinations = [
        (sampled_countries[0], sampled_oss[0]),
        (sampled_countries[0], sampled_oss[1]),
        (sampled_countries[1], sampled_oss[0]),
        (sampled_countries[1], sampled_oss[1])
    ]
    
    # Create the 2x2 subplot axes
    plot_axes = [
        fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    ]
    
    # Loop through combinations and plot
    for ax, (country, os_type) in zip(plot_axes, combinations):
        
        # 1. Get Model Prediction
        preds = simulate_demographic_creative(model, image_tensor, country, os_type, seq_len=50, device=device)
        days = range(1, len(preds) + 1)
        ax.plot(days, preds, color='blue', linestyle='--', marker='x', markersize=4, label="RNN Prediction")
        
        # 2. Get Ground Truth (if it exists for this exact cross-section)
        truth = agg_stats[
            (agg_stats['creative_id'] == sampled_cid) & 
            (agg_stats['country'] == country) & 
            (agg_stats['os'] == os_type)
        ].sort_values('days_since_launch')
        
        if not truth.empty:
            ax.plot(truth['days_since_launch'], truth['CTR'], color='green', marker='s', markersize=5, linewidth=2, label="Actual Truth")
            
        ax.set_title(f"Target: {country} | {os_type}")
        ax.set_xlabel("Days Since Launch")
        ax.set_ylabel("CTR")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()