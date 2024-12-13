from google.cloud import aiplatform
import os
from typing import Tuple

class DeploymentService:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.staging_bucket = f"gs://{os.environ.get('GCS_BUCKET_NAME')}"
        aiplatform.init(project=project_id, location=location, staging_bucket=self.staging_bucket)
        self.VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20241212_0916_RC00"

    def deploy_model(self, model_id: str, machine_type: str = "n1-standard-2", min_replica_count: int = 1, max_replica_count: int = 1) -> str:
        """Deploys a model to a Vertex AI Endpoint."""
        endpoint = aiplatform.Endpoint.create(
            display_name="llm-endpoint",
            project=self.project_id,
            location=self.location,
            create_request_timeout=180,
        )

        model = aiplatform.Model(model_name=model_id)

        deployed_model = model.deploy(
            endpoint=endpoint,
            deployed_model_display_name="deployed-llm",
            traffic_percentage=100,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            sync=False
        )

        return endpoint.name

    def get_deployment_status(self, endpoint_id: str):
        """Gets the status of a model deployment."""
        api_endpoint = f"{self.location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)
        
        endpoint_name = client.endpoint_path(
            project=self.project_id, location=self.location, endpoint=endpoint_id
        )
        
        response = client.get_endpoint(name=endpoint_name)

        state = "UNKNOWN"
        error_message = None
        for deployed_model in response.deployed_models:
            if deployed_model.display_name == "deployed-llm":
                if deployed_model.private_endpoints:
                    # For models using PrivateEndpoints
                    state = "DEPLOYED" if deployed_model.private_endpoints.predict_http_uri else "UNKNOWN"
                else:
                    # For models using PublicEndpoints
                    state = "DEPLOYED" if deployed_model.service_account else "UNKNOWN"
                break

        return {
            "endpoint_id": endpoint_id,
            "state": state,
            "error": error_message,
        }

    def deploy_model_vllm(
        self,
        model_name: str,
        model_id: str,
        service_account: str,
        machine_type: str = "g2-standard-8",
        accelerator_type: str = "NVIDIA_L4",
        accelerator_count: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        dtype: str = "auto",
        enable_trust_remote_code: bool = False,
        enforce_eager: bool = False,
        enable_lora: bool = True,
        max_loras: int = 1,
        max_cpu_loras: int = 8,
        use_dedicated_endpoint: bool = False,
        max_num_seqs: int = 256,
        model_type: str = None,
    ) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
        """Deploys trained models with vLLM into Vertex AI."""
        endpoint = aiplatform.Endpoint.create(
            display_name=f"{model_name}-endpoint",
            project=self.project_id,
            location=self.location,
        )
        # deploying vLLM model 
        print("deploying vLLM model in project", self.project_id, "location", self.location)

        vllm_args = [
            "python",
            "-m",
            "vllm.entrypoints.api_server",
            "--host=0.0.0.0",
            "--port=8080",
            f"--model={model_id}",
            f"--tensor-parallel-size={accelerator_count}",
            "--swap-space=16",
            f"--gpu-memory-utilization={gpu_memory_utilization}",
            f"--max-model-len={max_model_len}",
            f"--dtype={dtype}",
            f"--max-loras={max_loras}",
            f"--max-cpu-loras={max_cpu_loras}",
            f"--max-num-seqs={max_num_seqs}",
            "--disable-log-stats",
        ]

        if enable_trust_remote_code:
            vllm_args.append("--trust-remote-code")

        if enforce_eager:
            vllm_args.append("--enforce-eager")
        
        if enable_lora:
            print("enabling lora")
            vllm_args.append("--enable-lora")
        
        if model_type:
            vllm_args.append(f"--model-type={model_type}")

        
        hf_token =  os.environ.get("HF_TOKEN")


        env_vars = {
            "MODEL_ID": model_id,
            "DEPLOY_SOURCE": "notebook",
            "HF_TOKEN": hf_token
        }
        print(env_vars)
        model = aiplatform.Model.upload(
            display_name=model_name,
            serving_container_image_uri=self.VLLM_DOCKER_URI,
            serving_container_args=vllm_args,
            serving_container_ports=[8080],
            serving_container_predict_route="/generate",
            serving_container_health_route="/ping",
            serving_container_environment_variables=env_vars,
            serving_container_shared_memory_size_mb=(16 * 1024),
            serving_container_deployment_timeout=7200,
        )

        print(
            f"Deploying {model_name} on {machine_type} with {accelerator_count} {accelerator_type} GPU(s)."
        )

        model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            deploy_request_timeout=1800,
            service_account=service_account,
            sync=False
        )
        print("endpoint_name:", endpoint.name)

        return model, endpoint