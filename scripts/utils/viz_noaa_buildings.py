import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

def download_file(s3_client, bucket_name, key, local_path):
    """Worker function to download a single file."""
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Skip if it's a directory marker
    if key.endswith('/'):
        return

    try:
        print(f"Downloading: {key}")
        s3_client.download_file(bucket_name, key, local_path)
    except Exception as e:
        print(f"Error downloading {key}: {e}")

def download_melissa_parallel(local_dir="data/melissa_imagery", max_workers=10):
    bucket_name = "noaa-eri-pds"
    prefix = "2025_Hurricane_Melissa/"
    
    # Configure S3 client for anonymous access
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    print(f"Listing files from s3://{bucket_name}/{prefix}...")
    
    # Gather all keys first
    all_files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                relative_path = os.path.relpath(key, prefix)
                local_path = os.path.join(local_dir, relative_path)
                all_files.append((key, local_path))

    print(f"Found {len(all_files)} files. Starting parallel download with {max_workers} workers...")

    # Use ThreadPoolExecutor for I/O bound parallel tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, local_path in all_files:
            executor.submit(download_file, s3_client, bucket_name, key, local_path)

if __name__ == "__main__":
    # You can increase max_workers based on your network bandwidth
    download_melissa_parallel(max_workers=15)