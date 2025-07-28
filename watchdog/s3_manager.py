import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")  # Optional for MinIO

# Create the S3 client with optional endpoint URL
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def upload_file(local_path, s3_key):
    """Upload a file to S3 bucket."""
    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"✅ Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

def download_file(s3_key, local_path):
    """Download a file from S3 bucket."""
    try:
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
        print(f"✅ Downloaded s3://{S3_BUCKET}/{s3_key} to {local_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")

def list_files(prefix=""):
    """List files in S3 bucket with a given prefix."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        for obj in response.get('Contents', []):
            print(obj['Key'])
    except Exception as e:
        print(f"❌ Listing failed: {e}")
