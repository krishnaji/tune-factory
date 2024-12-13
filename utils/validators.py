def validate_gcs_url(gcs_url: str, bucket_name: str):
    if not gcs_url.startswith(f"gs://{bucket_name}/"):
        raise ValueError("Invalid GCS URL. Must start with 'gs://'")