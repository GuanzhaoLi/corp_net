import os
from cropnet.data_downloader import DataDownloader

# User provided credentials
CLIENT_ID = 'f65bc524-5da5-4fa0-b185-0ec23363c9fb'
CLIENT_SECRET = 'q2tahokQ50QT0aUdnBS0iLrhI1ND90wj'

def main():
    target_dir = "./demo_data"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    print(f"Initializing DataDownloader to {target_dir}...")
    # Initialize with user credentials
    downloader = DataDownloader(
        target_dir=target_dir,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    
    # Champaign County, IL
    fips_codes = ["17019"]
    years = ["2022"]
    
    print("\n--- Downloading Sentinel-2 Imagery ---")
    try:
        # Download Agricultural (AG) images
        downloader.download_Sentinel2(fips_codes=fips_codes, years=years, image_type="AG")
    except Exception as e:
        print(f"Error downloading Sentinel-2 data: {e}")

if __name__ == "__main__":
    main()
