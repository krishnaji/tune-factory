import os
from services.gcs_service import GcsService
from services.training_service import TrainingService
from services.deployment_service import DeploymentService

# Initialize GCS bucket name, project id from environment variables
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")  # Default to "us-central1"
MODEL_IMAGE_URI = os.environ.get(
    "MODEL_IMAGE_URI",
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/llamafactory/llama-factory:latest",
)
HF_TOKEN = os.environ.get("HF_TOKEN")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

def get_gcs_service() -> GcsService:
    return GcsService(bucket_name=GCS_BUCKET_NAME)

def get_training_service() -> TrainingService:
    return TrainingService(
        project_id=PROJECT_ID,
        location=LOCATION,
        model_image_uri=MODEL_IMAGE_URI,
        hf_token=HF_TOKEN,
    )

def get_deployment_service() -> DeploymentService:
    return DeploymentService(project_id=PROJECT_ID, location=LOCATION)

# print all dependecies
print(f"GCS_BUCKET_NAME: {GCS_BUCKET_NAME}")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"LOCATION: {LOCATION}")
print(f"MODEL_IMAGE_URI: {MODEL_IMAGE_URI}")
print(f"HF_TOKEN: {HF_TOKEN}")
print(f"SERVICE_ACCOUNT: {SERVICE_ACCOUNT}")
