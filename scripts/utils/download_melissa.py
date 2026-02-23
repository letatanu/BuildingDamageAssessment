import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os

# Configure boto3 to access public S3 bucket without credentials
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Bucket and folder configuration
bucket_name = 'noaa-eri-pds'
folder_prefix = '2025_Hurricane_Melissa/20251106b_RGB/'
download_dir = 'data/hurricane_melissa_tif/20251106b_RGB'

# Create download directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# List all objects in the folder
paginator = s3_client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)

tif_count = 0
for page in pages:
    if 'Contents' not in page:
        continue
    
    for obj in page['Contents']:
        key = obj['Key']
        
        # Download only .tif files (skip .geom, .jpg, and folders)
        if key.endswith('.tif'):
            filename = os.path.basename(key)
            local_path = os.path.join(download_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(local_path):
                print(f"Skipping {filename} (already exists)")
                continue
            
            print(f"Downloading {filename} ({obj['Size'] / 1024 / 1024:.1f} MB)...")
            s3_client.download_file(bucket_name, key, local_path)
            tif_count += 1
            print(f"✓ Downloaded: {filename}")

print(f"\nTotal .tif files downloaded: {tif_count}")
