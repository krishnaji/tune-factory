from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from services.gcs_service import GcsService
from schemas import DatasetItem
from typing import List, Optional, Dict
import os
import json



router = APIRouter(prefix="/datasets", tags=["Datasets"])

DATASET_INFO_FILE = "dataset_info.json" 

@router.post("/upload", response_model=dict)
async def upload_dataset(
    file: UploadFile = File(...),
    gcs_service: GcsService = Depends(GcsService),
    formatting: Optional[str] = None,
    columns: Optional[Dict[str, str]] = None,
):
    """Uploads a dataset to GCS and updates dataset_info.json."""
    try:
        # Upload the dataset file to GCS
        destination_blob_name = f"datasets/{file.filename}"
        gcs_url = gcs_service.upload_file(file.file, destination_blob_name)

        # Update dataset_info.json
        dataset_name = os.path.splitext(file.filename)[0]
        await update_dataset_info(
            gcs_service, dataset_name, file.filename, formatting, columns
        )

        return {"message": f"Dataset uploaded to {gcs_url}", "gcs_url": gcs_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def update_dataset_info(
    gcs_service: GcsService,
    dataset_name: str,
    file_name: str,
    formatting: Optional[str] = None,
    columns: Optional[Dict[str, str]] = None,
):
    """Creates or updates dataset_info.json with the new dataset information."""
    local_info_file = "dataset_info.json"
    remote_info_path = f"datasets/{DATASET_INFO_FILE}"
    print(f"Attempting to download dataset_info from: {remote_info_path}")
    # Download dataset_info.json if it exists
    try:
        dataset_info_content = gcs_service.download_file(
           remote_info_path
        )
        print(f"Successfully downloaded dataset_info.json")
        with open(local_info_file, "wb") as f:
            f.write(dataset_info_content)
        with open(local_info_file, "r") as f:
            dataset_info = json.load(f)
    except FileNotFoundError:
        print(f"dataset_info.json not found, creating a new one.")
        dataset_info = {}
    except Exception as e:
        print(f"Error during downloading dataset_info: {e}")
        dataset_info = {}

    # Update the dataset information
    dataset_info[dataset_name] = {"file_name": file_name}
    if formatting:
        dataset_info[dataset_name]["formatting"] = formatting
    if columns:
        dataset_info[dataset_name]["columns"] = columns

    # Upload the updated dataset_info.json
    with open(local_info_file, "w") as f:
        json.dump(dataset_info, f)
    with open(local_info_file, "rb") as f:
        print(f"Attempting to upload updated dataset_info to: {remote_info_path}")
        gcs_service.upload_file(f, remote_info_path)
        print(f"Successfully uploaded dataset_info.json")

@router.get("/", response_model=List[DatasetItem])
async def list_datasets(gcs_service: GcsService = Depends(GcsService)):
    """Lists all uploaded datasets."""
    try:
        files = gcs_service.list_files("datasets/")
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{gcs_url:path}", response_model=bytes)
async def get_dataset(gcs_url: str, gcs_service: GcsService = Depends(GcsService)):
    """Retrieves a specific dataset."""
    try:
        file_content = gcs_service.download_file(gcs_url)
        return file_content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid GCS URL")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))