import h5py
import os

# Path to one of the downloaded files
h5_path = "/Users/guanzhaoli/PycharmProjects/corp_net/demo_data/Sentinel/data/AG/2022/IL/Agriculture_17_IL_2022-01-01_2022-03-31.h5"

if not os.path.exists(h5_path):
    print(f"File not found: {h5_path}")
    # Try to find any h5 file in that dir
    base_dir = "/Users/guanzhaoli/PycharmProjects/corp_net/demo_data/Sentinel/data/AG/2022/IL/"
    if os.path.exists(base_dir):
        files = os.listdir(base_dir)
        h5_files = [f for f in files if f.endswith('.h5')]
        if h5_files:
            h5_path = os.path.join(base_dir, h5_files[0])
            print(f"Using alternative file: {h5_path}")
        else:
            print("No H5 files found.")
            exit()
    else:
        print(f"Directory not found: {base_dir}")
        exit()

print(f"Inspecting: {h5_path}")

try:
    with h5py.File(h5_path, 'r') as f:
        print("Keys:", list(f.keys()))
        
        # Recursively print structure for first few items
        def visit_func(name, node):
            print(f"Name: {name}")
            if isinstance(node, h5py.Dataset):
                print(f"  Shape: {node.shape}, Dtype: {node.dtype}")
                # Don't print content, just metadata
            elif isinstance(node, h5py.Group):
                print(f"  Group keys: {list(node.keys())[:5]}...") # Print first 5 keys
                
        # Visit first level
        for key in list(f.keys())[:5]:
            print(f"\n--- {key} ---")
            item = f[key]
            if isinstance(item, h5py.Group):
                 # Just peek inside
                 print(f"Group with keys: {list(item.keys())[:5]}...")
                 # Check one valid fips inside if possible
                 if len(item.keys()) > 0:
                     sub_key = list(item.keys())[0]
                     print(f"  Sample Item {sub_key}: {item[sub_key]}")
                     if isinstance(item[sub_key], h5py.Dataset):
                         print(f"  Sample Shape: {item[sub_key].shape}")

            elif isinstance(item, h5py.Dataset):
                print(f"Dataset: {item.shape}")

except Exception as e:
    print(f"Error inspecting H5: {e}")
