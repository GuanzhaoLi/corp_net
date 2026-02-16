import h5py
import os

# Path to one of the downloaded files
h5_path = "/Users/guanzhaoli/PycharmProjects/corp_net/demo_data/Sentinel/data/AG/2022/IL/Agriculture_17_IL_2022-01-01_2022-03-31.h5"

print(f"Inspecting deep structure of: {h5_path}")

try:
    with h5py.File(h5_path, 'r') as f:
        # We know 17019 is a key
        if '17019' in f:
            fips_group = f['17019']
            print("FIPS Group keys:", list(fips_group.keys())[:5])
            
            # Pick first date
            first_date = list(fips_group.keys())[0]
            date_group = fips_group[first_date]
            print(f"--- Date Group: {first_date} ---")
            print("Keys (Bands?):", list(date_group.keys()))
            
            for key in date_group.keys():
                item = date_group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  Start Dataset '{key}': Shape={item.shape}, Dtype={item.dtype}")
        else:
            print("17019 not found in file.")

except Exception as e:
    print(f"Error: {e}")
