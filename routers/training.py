from fastapi import APIRouter, HTTPException, Depends
from services.gcs_service import GcsService
from services.training_service import TrainingService
from schemas import GenerateConfigSchema, StartTrainingSchema, TrainingConfigItem, TrainingJobStatus
from utils.config_generator import generate_training_config
from dependencies import get_training_service, get_gcs_service 
from typing import List
import os

router = APIRouter(prefix="/training", tags=["Training"])

@router.post("/generate_config", response_model=dict)
async def generate_config_route(
    config_data: GenerateConfigSchema,
    gcs_service: GcsService = Depends(GcsService)
):
    """Generates LLM training YAML configuration."""
    try:
        yaml_content = generate_training_config(
            config_data.dataset_dir, config_data.model_name_or_path, config_data.output_dir, config_data.dataset, config_data.training_config
        )
        destination_blob_name = f"training_configs/training_config_{os.urandom(4).hex()}.yaml"
        gcs_url = gcs_service.upload_string_as_file(yaml_content, destination_blob_name)
        return {
            "message": f"Training YAML uploaded to {gcs_url}",
            "gcs_url": gcs_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/configs", response_model=List[TrainingConfigItem])
async def list_training_configs(gcs_service: GcsService = Depends(GcsService)):
    """Lists all training YAML configurations."""
    try:
        files = gcs_service.list_files("training_configs/")
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start", response_model=dict)
async def start_training(
    training_data: StartTrainingSchema,
    gcs_service: GcsService = Depends(GcsService),
    # training_service: TrainingService = Depends(TrainingService)
    training_service: TrainingService = Depends(get_training_service)
):
    """Starts a Vertex AI Custom Training job."""
    try:
        if not gcs_service.file_exists(training_data.config_gcs_url):
            raise HTTPException(status_code=400, detail=f"Training config not found: {training_data.config_gcs_url}")
        job_id = training_service.start_training_job(training_data.config_gcs_url)
        return {"message": "Vertex AI Custom Training job submitted", "job_id": job_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid GCS URL")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=TrainingJobStatus)
async def get_training_status(job_id: str,  training_service: TrainingService = Depends(get_training_service)):
    """Checks the status of a training job."""
    try:
        status = training_service.get_training_job_status(job_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))