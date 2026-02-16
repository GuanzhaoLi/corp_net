import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CropYieldDataset
from model import CropYieldModel
from config import Config
import os

def train():
    # 1. Setup
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data
    dataset = CropYieldDataset(
        root_dir=config.ROOT_DIR,
        years=config.YEARS,
        fips_codes=config.FIPS_CODES,
        crop_type=config.CROP_TYPE
    )
    
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 3. Model
    model = CropYieldModel(config).to(device)
    
    # 4. Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 5. Loop
    model.train()
    print("Starting training...")
    
    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(device) # (B, T, C, H, W)
            targets = batch['yield'].to(device) # (B, 1)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(images) # (B,)
            
            # Loss
            loss = criterion(outputs, targets.squeeze(-1))
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 1 == 0:
                print(f"Epoch [{epoch+1}/{config.EPOCHS}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Average Loss: {avg_loss:.4f}")
        
    # Save
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "model_latest.pth"))
    print("Model saved.")

if __name__ == "__main__":
    train()
