# Config for Crop Yield Prediction (ST-ViT: Spatial-Temporal ViT)

class Config:
    # Data Parameters
    ROOT_DIR = "./demo_data"
    YEARS = ["2025"]
    # FIPS used for training (default: same as US_ESTIMATE_FIPS for retrain with NASS yields)
    FIPS_CODES = [
        "17113",
        "18011",
        "26063",
        "27129",
        "29153",
        "28011",
        "38017",
        "31019",
        "39037",
        "46137",
        "55051",
    ]
    # If set, training loads yield from this CSV (fips, year, actual_yield_bu_per_acre) instead of DataRetriever.
    YIELD_CSV = "fips_harvested_acres_example_actual_yield.csv"
    # Set True to retrain with 11 representative counties and 2016-2024 (NASS yields from CSV).
    TRAIN_WITH_NASS_CSV = True
    CROP_TYPE = "Soybean" # User downloaded Soybean data

    # US national estimate from 15 representative counties (predict_batch + aggregate_to_us)
    # Span: IL, IA, IN, MN, NE, OH, MS, ND, AR, SD, MO
    US_ESTIMATE_FIPS = [
        "17113",
        "18011",
        "26063",
        "27129",
        "29153",
        "28011",
        "38017",
        "31019",
        "39037",
        "46137",
        "55051",
    ]
    # US total soybean harvested acres by year (approx, NASS); replace with NASS for accuracy
    US_SOYBEAN_ACRES_2024 = 86.0e6
    US_SOYBEAN_ACRES_BY_YEAR = {
        2016: 83.0e6, 2017: 89.5e6, 2018: 88.2e6, 2019: 75.0e6, 2020: 83.4e6,
        2021: 86.4e6, 2022: 86.3e6, 2023: 86.2e6, 2024: 86.0e6, 2025: 86.0e6,
    }  # 2025: placeholder (use NASS when available)

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
    LEARNING_RATE = 5e-5  # Slightly lower can help avoid collapse to constant (was 1e-4)
    EPOCHS = 30  # More epochs with augmentation
    GRAD_CLIP = 1.0  # Gradient clipping for stability
    DEVICE = "cuda" # Will fallback to cpu/mps automatically
    
    # Output
    CHECKPOINT_DIR = "./checkpoints"
