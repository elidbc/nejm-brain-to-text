import h5py
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

def check_data_lengths():
    # Load your config to get paths
    with open('exp1_args.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    base_dir = config['dataset']['dataset_dir']
    sessions = config['dataset']['sessions']
    
    all_lengths = []
    
    print(f"Scanning {len(sessions)} sessions...")
    
    for session in sessions:
        path = os.path.join(base_dir, session, 'data_train.hdf5')
        if not os.path.exists(path):
            continue
            
        with h5py.File(path, 'r') as f:
            # Iterate over all trials in the file
            for key in f.keys():
                # n_time_steps is the raw neural length
                if 'n_time_steps' in f[key].attrs:
                    all_lengths.append(f[key].attrs['n_time_steps'])
    
    all_lengths = np.array(all_lengths)
    print("\n=== Data Length Statistics (20ms bins) ===")
    print(f"Min: {np.min(all_lengths)}")
    print(f"Mean: {np.mean(all_lengths):.2f}")
    print(f"Max: {np.max(all_lengths)}")
    print(f"99th Percentile: {np.percentile(all_lengths, 99):.2f}")
    
    # Check theoretical memory for Batch=16 at Max Length
    # Patching reduces time by 4, but increases dim to 7168
    # Float32 = 4 bytes
    max_len = np.max(all_lengths)
    patched_len = max_len / 4
    input_tensor_size = 16 * patched_len * 7168 * 4 / (1024**3) # in GB
    
    print(f"\nAt Batch Size 16 and Max Length {max_len}:")
    print(f"The GRU Input Tensor alone would be: {input_tensor_size:.2f} GB")
    print("(Total training memory will be ~5-10x this value)")

if __name__ == "__main__":
    check_data_lengths()