import os
from cropnet.data_downloader import DataDownloader

from config import Config

# User provided credentials
CLIENT_ID = 'f65bc524-5da5-4fa0-b185-0ec23363c9fb'
CLIENT_SECRET = 'q2tahokQ50QT0aUdnBS0iLrhI1ND90wj'

def main():
    config = Config()
    target_dir = config.ROOT_DIR
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    print(f"Initializing DataDownloader to {target_dir}...")
    # Initialize with user credentials
    downloader = DataDownloader(
        target_dir=target_dir,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    
    # Use Config
    fips_codes = config.FIPS_CODES
    years = config.YEARS
    
    print(f"\n--- Downloading Sentinel-2 Imagery for {len(fips_codes)} counties & {len(years)} years ---")
    print(f"Counties: {fips_codes}")
    print(f"Years: {years}")

    try:
        # Download Agricultural (AG) images
        downloader.download_Sentinel2(fips_codes=fips_codes, years=years, image_type="AG")
    except Exception as e:
        print(f"Error downloading Sentinel-2 data: {e}")


if __name__ == "__main__":
    main()
