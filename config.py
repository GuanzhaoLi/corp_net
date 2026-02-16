# Config for Crop Yield Prediction

class Config:
    # Data Parameters
    ROOT_DIR = "./demo_data"
    YEARS = ["2020", "2021", "2022"]
    # Top 10 IL Corn Counties (2022)
    FIPS_CODES = [
        "17113", # McLean
        "18093", # Lawrence
        "19047", # Crawford
        "23009", # Hancock
        "27021", # Cass
    ]
    CROP_TYPE = "Soybean" # User downloaded Soybean data
    
    # Image Parameters
    IMG_H = 224
    IMG_W = 224
    CHANNELS = 3
    
    # Model Hyperparameters
    VISUAL_BACKBONE = "vit_b_16" # "resnet18" or "vit_b_16"
    HIDDEN_DIM = 768 # ViT-B usually outputs 768
    
    # Temporal Transformer params
    TEMPORAL_LAYERS = 2
    TEMPORAL_HEADS = 4
    DROPOUT = 0.1
    
    # Training Parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4 # ViT might need smaller LR
    EPOCHS = 1 # Quick test
    DEVICE = "cuda" # Will fallback to cpu/mps automatically
    
    # Output
    CHECKPOINT_DIR = "./checkpoints"
