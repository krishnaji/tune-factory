from google.cloud import storage
from typing import List, Optional, BinaryIO, Dict
import os
from utils.validators import validate_gcs_url

class GcsService:
    def __init__(self, bucket_name: str = None):
        self.client = storage.Client()
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
        if self.bucket_name:
            self.bucket = self.client.bucket(self.bucket_name)
        else:
            raise ValueError("GCS_BUCKET_NAME environment variable not set.")

    def upload_file(self, file: BinaryIO, destination_blob_name: str) -> str:
        """Uploads a file to the bucket."""
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_file(file)
        return f"gs://{self.bucket_name}/{destination_blob_name}"
    
    def upload_string_as_file(self, content: str, destination_blob_name: str) -> str:
        """Uploads a string as a file to the bucket."""
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(content)
        return f"gs://{self.bucket_name}/{destination_blob_name}"

    def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, str]]:
        """Lists all the files in the bucket."""
        blobs = self.bucket.list_blobs(prefix=prefix)
        files = []
        for blob in blobs:
            files.append({
                "filepath": blob.name,
                "gcs_url": f"gs://{self.bucket_name}/{blob.name}"
            })
        return files

    def download_file(self, gcs_url: str) -> BinaryIO:
        """Downloads a file from the bucket."""
        validate_gcs_url(gcs_url, self.bucket_name)
        blob = self.bucket.blob(gcs_url.replace(f"gs://{self.bucket_name}/", ""))
        
        if not blob.exists():
            raise FileNotFoundError(f"File not found: {gcs_url}")
        return blob.download_as_bytes()
    
    def file_exists(self, gcs_url: str) -> bool:
        """Checks if a file exists in the bucket."""
        validate_gcs_url(gcs_url, self.bucket_name)
        blob = self.bucket.blob(gcs_url.replace(f"gs://{self.bucket_name}/", ""))
        return blob.exists()