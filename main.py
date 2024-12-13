from fastapi import FastAPI, Depends
from routers import datasets, training, deployment
from services.gcs_service import GcsService
from services.training_service import TrainingService
from services.deployment_service import DeploymentService
import os
from dotenv import load_dotenv
from dependencies import get_gcs_service, get_training_service, get_deployment_service 

load_dotenv()

app = FastAPI(
    title="LLM Training and Deployment API",
    description="API for managing datasets, training LLMs, and deploying them on Vertex AI",
    version="v1",
)

# Inject dependencies into routers
app.include_router(
    datasets.router, dependencies=[Depends(get_gcs_service)]
)
app.include_router(
    training.router,
    dependencies=[Depends(get_gcs_service), Depends(get_training_service)],
)
app.include_router(
    deployment.router, dependencies=[Depends(get_deployment_service)]
)