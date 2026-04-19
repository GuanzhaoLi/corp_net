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
        
    def forward(self, x, key_padding_mask=None):
        # x: (B, T, D). key_padding_mask: (B, T) True = ignore (e.g. padded)
        B, T, D = x.shape
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1) # (B, T+1, D)
        
        # Mask for transformer: (B, T+1), True = ignore. CLS at 0 is never masked.
        if key_padding_mask is not None:
            # key_padding_mask is (B, T) for sequence; prepend False for CLS
            pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), key_padding_mask], dim=1)
        else:
            pad_mask = None
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Pass
        out = self.transformer_encoder(x, src_key_padding_mask=pad_mask) # (B, T+1, D)
        
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
        
        # Stronger head with dropout to force use of features and reduce constant collapse
        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 1),
        )
        
    def forward(self, x, lengths=None):
        # x: (B, T, C, H, W). lengths: (B,) actual time steps per sample (optional, for variable T)
        B, T, C, H, W = x.shape
        
        # Flatten B and T to pass through Visual Encoder
        x_flat = x.contiguous().reshape(B * T, C, H, W)
        
        # Visual Feats (Spatial ViT)
        visual_feats = self.visual_encoder(x_flat) # (B*T, D)
        
        # Reshape back to Sequence
        visual_feats = visual_feats.reshape(B, T, -1) # (B, T, D)
        
        # Mask for padded time steps (True = ignore)
        if lengths is not None:
            # lengths: (B,) e.g. [12, 10, 15, 12]; mask[b, t] = True when t >= lengths[b]
            key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T)
        else:
            key_padding_mask = None
        
        # Temporal Processing (Temporal Transformer)
        context_vec = self.temporal_encoder(visual_feats, key_padding_mask=key_padding_mask) # (B, D)
        
        # Prediction
        prediction = self.head(context_vec) # (B, 1)
        
        return prediction.squeeze(-1)


class MacroEncoder(nn.Module):
    """
    Encodes yearly macroeconomic data (CPI, Interest Rates, Fertilizer Prices, etc.)
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.macro_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        # x: (B, input_dim)
        return self.macro_encoder(x)

class CropPriceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.monthly_national_macro = bool(getattr(config, "MONTHLY_NATIONAL_MACRO", False))

        # Visual/Temporal Branch (ViT + Temporal Transformer on satellite image sequence)
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

        # Per-frame national macro (monthly row aligned to satellite date) -> same dim as visual tokens
        if self.monthly_national_macro:
            self.macro_time_embed = nn.Sequential(
                nn.Linear(config.MACRO_INPUT_DIM, config.HIDDEN_DIM),
                nn.LayerNorm(config.HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(config.DROPOUT),
            )

        # Macro Branch
        self.macro_encoder = MacroEncoder(
            input_dim=config.MACRO_INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT
        )

        # Gated Fusion Layer: concat visual + macro (2 * D) -> per-dim gate in [0, 1]
        self.gated_fusion = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.Sigmoid(),
        )

        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 1) # Predicting a single scalar: price basis
        )

    def forward(self, images, macro_data, lengths=None):
        """
        Args:
            images: (B, T, C, H, W) - Sentinel 2 satellite image sequence
            macro_data: (B, M) yearly macro, or (B, T, M) national monthly macro aligned to each frame
            lengths: (B,) - Actual image sequence lengths for masking
        """
        # x: (B, T, C, H, W). lengths: (B,) actual time steps per sample (optional, for variable T)
        B, T, C, H, W = images.shape

        # Visual/Temporal Branch (ViT + Temporal Transformer on satellite image sequence)
        # Flatten B and T for the ViT/ResNet backbone
        images_flat = images.contiguous().reshape(B * T, C, H, W)
        # Visual Feats (Spatial ViT)
        visual_feats = self.visual_encoder(images_flat) # (B * T, D)
        # Reshape back to Sequence
        visual_feats = visual_feats.reshape(B, T, -1) # (B, T, D)

        # Mask for padded time steps (True = ignore)
        if lengths is not None:
            # lengths: (B,) e.g. [12, 10, 15, 12]; mask[b, t] = True when t >= lengths[b]
            key_padding_mask = torch.arange(T, device=images.device).unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T)
        else:
            key_padding_mask = None

        if macro_data.dim() == 3:
            if not self.monthly_national_macro:
                raise ValueError("macro_data is (B, T, M) but config.MONTHLY_NATIONAL_MACRO is False")
            macro_emb = self.macro_time_embed(macro_data)
            visual_feats = visual_feats + macro_emb
            if lengths is not None:
                valid = ~key_padding_mask
                denom = valid.sum(dim=1, keepdim=True).to(macro_data.dtype).clamp(min=1.0)
                macro_pooled = (macro_data * valid.unsqueeze(-1)).sum(dim=1) / denom
            else:
                macro_pooled = macro_data.mean(dim=1)
            macro_vec = self.macro_encoder(macro_pooled)
        else:
            macro_vec = self.macro_encoder(macro_data)
        
        # Temporal Processing (Temporal Transformer)
        context_vec = self.temporal_encoder(visual_feats, key_padding_mask=key_padding_mask) # (B, D)

        # Gated Fusion Layer
        # Concatenate all features
        combined_vec = torch.cat([context_vec, macro_vec], dim=-1) # (B, D * 2)
        gate_weight = self.gated_fusion(combined_vec) # (B, D)
        
        # Compute an element-wise weight (gate) between 0 and 1
        # If gate is high, visual features are prioritized; if low, macro is prioritized
        fused = gate_weight * context_vec + (1 - gate_weight) * macro_vec

        # Prediction
        price_basis_prediction = self.head(fused) # (B, 1)

        return price_basis_prediction.squeeze(-1)