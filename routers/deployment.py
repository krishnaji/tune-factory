from fastapi import APIRouter, HTTPException, Depends
from services.deployment_service import DeploymentService
from schemas import DeployModelSchema, DeploymentJobStatus, VLLMDeployModelSchema
from typing import Tuple
from dependencies import get_deployment_service

router = APIRouter(prefix="/deployment", tags=["Deployment"])

@router.post("/deploy", response_model=dict)
async def deploy_model(
    deployment_data: DeployModelSchema,
    deployment_service: DeploymentService = Depends(DeploymentService),
    machine_type: str = "n1-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = 1
):
    """Deploys a trained model as a Vertex AI Endpoint."""
    try:
        endpoint_id = deployment_service.deploy_model(deployment_data.model_id, machine_type, min_replica_count, max_replica_count)
        return {
            "message": "Vertex AI Endpoint deployment job submitted",
            "endpoint_id": endpoint_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{endpoint_id}", response_model=DeploymentJobStatus)
async def get_deployment_status(endpoint_id: str, 
                                deployment_service: DeploymentService = Depends(get_deployment_service)):
    """Checks the status of a model deployment job."""
    try:
        status = deployment_service.get_deployment_status(endpoint_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deploy_vllm", response_model=dict)
async def deploy_vllm_model(
    deployment_data: VLLMDeployModelSchema,
    deployment_service: DeploymentService = Depends(get_deployment_service) 
) -> dict:
    """Deploys a trained model with vLLM into Vertex AI."""
    try:
        _, endpoint = deployment_service.deploy_model_vllm(
            model_name=deployment_data.model_name,
            model_id=deployment_data.model_id,
            service_account=deployment_data.service_account,
            machine_type=deployment_data.machine_type,
            accelerator_type=deployment_data.accelerator_type,
            accelerator_count=deployment_data.accelerator_count,
            gpu_memory_utilization=deployment_data.gpu_memory_utilization,
            max_model_len=deployment_data.max_model_len,
            dtype=deployment_data.dtype,
            enable_trust_remote_code=deployment_data.enable_trust_remote_code,
            enforce_eager=deployment_data.enforce_eager,
            enable_lora=deployment_data.enable_lora,
            max_loras=deployment_data.max_loras,
            max_cpu_loras=deployment_data.max_cpu_loras,
            use_dedicated_endpoint=deployment_data.use_dedicated_endpoint,
            max_num_seqs=deployment_data.max_num_seqs,
            model_type=deployment_data.model_type,
        )
        return {
            "message": "Vertex AI Endpoint deployment job submitted for vLLM model",
            "endpoint_id": endpoint.name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))