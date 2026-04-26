import torch
import torch.nn as nn
import torchvision.models as models
import random

class UNetEncoder(nn.Module):
    """
    Extracts latent features using the encoder part of a U-Net architecture.
    Here we use a standard ResNet backbone, which is a common and robust U-Net encoder.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        # Use ResNet18 as the U-Net contracting path (encoder)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.latent_proj = nn.Linear(resnet.fc.in_features, latent_dim)
        
        # Freeze the U-Net encoder parameters as requested
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad(): # Ensure encoder remains frozen during forward pass
            features = self.encoder(x)
        features = features.view(features.size(0), -1)
        # Project to our desired latent dimensionality
        return self.latent_proj(features)
    
class ImageAutoregressiveRNN(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=64, num_countries=10, num_os=2):
        super().__init__()
        
        # 1. Image Encoder
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, latent_dim)
        self.encoder = resnet
        
        # 2. INCREASED Embedding sizes
        # Giving categorical variables more "weight" against the 128-dim image
        self.country_emb = nn.Embedding(num_embeddings=num_countries, embedding_dim=16)
        self.os_emb = nn.Embedding(num_embeddings=num_os, embedding_dim=8)
        
        # 3. NEW: The Feature Fusion Layer
        # Image (128) + Country (16) + OS (8) = 152
        # This forces the model to learn interactions between the image and the demographics
        self.fusion = nn.Sequential(
            nn.Linear(152, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 4. Update RNN Input Size
        # Fused Context (64) + Prev_Y (1) = 65
        self.rnn = nn.GRU(input_size=65, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, images, country_idx, os_idx, targets=None, seq_len=30, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)
        device = images.device
        
        # Extract features
        latents = self.encoder(images)  # [batch_size, 128]
        c_emb = self.country_emb(country_idx) # [batch_size, 16]
        o_emb = self.os_emb(os_idx)           # [batch_size, 8]
        
        # Combine and FUSE the static context
        raw_context = torch.cat([latents, c_emb, o_emb], dim=1) # [batch_size, 152]
        fused_context = self.fusion(raw_context)                # [batch_size, 64]
        
        outputs = []
        hidden = None
        
        # Start Token for Day 0
        prev_y = torch.full((batch_size, 1), -1.0).to(device)
        
        active_seq_len = targets.size(1) if targets is not None else seq_len
        
        for t in range(active_seq_len):
            # Concatenate the fused context with the previous CTR
            rnn_in = torch.cat([fused_context, prev_y], dim=1).unsqueeze(1) # [batch_size, 1, 65]
            
            out, hidden = self.rnn(rnn_in, hidden)
            pred = self.fc(out.squeeze(1))
            outputs.append(pred)
            
            # Teacher forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_y = targets[:, t, :] 
            else:
                prev_y = pred
                
        return torch.stack(outputs, dim=1)