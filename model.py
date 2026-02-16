import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class VisualEncoder(nn.Module):
    def __init__(self, backbone_name="vit_b_16", pretrained=True, out_dim=768):
        super().__init__()
        
        if "vit" in backbone_name:
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            self.backbone.heads = nn.Identity() 
            self_feat_dim = 768 # ViT-B width
            
        elif backbone_name == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
            self_feat_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
            
        self.projector = nn.Linear(self_feat_dim, out_dim)
        
    def forward(self, x):
        # x: (B, C, H, W)
        features = self.backbone(x) # ViT: (B, 768) (after Identity head), ResNet: (B, 512, 1, 1)
        
        if len(features.shape) > 2:
            features = features.flatten(1)
            
        out = self.projector(features) # (B, out_dim)
        return out

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Classification token for temporal aggregation (Optional, or just use Average/Last)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1) # (B, T+1, D)
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Pass
        out = self.transformer_encoder(x) # (B, T+1, D)
        
        # Use CLS token output for prediction
        return out[:, 0, :]

class CropYieldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.visual_encoder = VisualEncoder(
            backbone_name=config.VISUAL_BACKBONE,
            pretrained=True, 
            out_dim=config.HIDDEN_DIM
        )
        
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=config.HIDDEN_DIM,
            num_layers=config.TEMPORAL_LAYERS,
            nhead=config.TEMPORAL_HEADS,
            dropout=config.DROPOUT
        )
        
        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Yield Prediction
        )
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Flatten B and T to pass through Visual Encoder
        # Use reshape to handle non-contiguous memory safely
        x_flat = x.contiguous().reshape(B * T, C, H, W)
        
        # Visual Feats (Spatial ViT)
        visual_feats = self.visual_encoder(x_flat) # (B*T, D)
        
        # Reshape back to Sequence
        visual_feats = visual_feats.reshape(B, T, -1) # (B, T, D)
        
        # Temporal Processing (Temporal Transformer)
        context_vec = self.temporal_encoder(visual_feats) # (B, D)
        
        # Prediction
        prediction = self.head(context_vec) # (B, 1)
        
        return prediction.squeeze(-1)
