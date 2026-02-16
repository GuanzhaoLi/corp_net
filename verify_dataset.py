from dataset import CropYieldDataset
import torch
import os

def test_dataset():
    print("--- Testing CropYieldDataset ---")
    
    # Setup paths
    root_dir = "./demo_data"
    years = ["2022"] 
    fips_codes = ["17019"] # Champaign, IL
    
    # Initialize implementation
    try:
        ds = CropYieldDataset(root_dir=root_dir, years=years, fips_codes=fips_codes, crop_type="Soybean")
        
        print(f"Dataset length: {len(ds)}")
        
        # Test __getitem__
        print("Fetching first sample...")
        sample = ds[0]
        
        images = sample['images']
        yield_target = sample['yield']
        fips = sample['fips']
        dates = sample['dates']
        
        print(f"FIPS: {fips}")
        print(f"Images Shape: {images.shape} (T, C, H, W)")
        print(f"Num Frames: {images.shape[0]}")
        print(f"Dates: {dates}")
        print(f"Yield Target: {yield_target.item()}")
        
        # Check integrity
        if images.shape[1] != 3:
            print("WARNING: Channels dim is not 3. Check data.")
        if images.shape[2] != 224 or images.shape[3] != 224:
            print("WARNING: Spatial dim is not 224x224.")
            
        print("Test Passed: Dataset works as expected.")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
