from google.cloud import aiplatform
import os

class TrainingService:
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_image_uri: str = None,
        hf_token: str = None,
    ):
        self.project_id = project_id
        self.location = location
        self.model_image_uri = model_image_uri
        self.hf_token = hf_token
        self.client = aiplatform.gapic.JobServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
        self.bucket_name = os.environ.get("GCS_BUCKET_NAME")
    def start_training_job(self, config_gcs_url: str) -> str:
        """Starts a Vertex AI Custom Training job."""

        # Extract the path relative to the bucket
        relative_path = config_gcs_url.replace(f"gs://{self.bucket_name}/", "")
        # Construct the /gcs/ path
        gcs_path = f"/gcs/{self.bucket_name}/{relative_path}"
        print("GCS Config Path:", gcs_path)
        print("HF_TOKEN:", self.hf_token)
        custom_job = {
            "display_name": "llm-training-job",
            "job_spec": {
                "worker_pool_specs": [
                    {
                        "machine_spec": {
                            "machine_type": "a2-highgpu-8g",
                            "accelerator_type":  getattr(aiplatform.gapic.AcceleratorType, "NVIDIA_TESLA_A100"), 
                            "accelerator_count": 8 
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": self.model_image_uri.format(
                                PROJECT_ID=self.project_id
                            ),
                            "command": [ "bash", 
                                        "-c",
                                        f"/usr/local/bin/llamafactory-cli train {gcs_path}"
                                        ],
                            "env": [
                                {"name": "HF_TOKEN", "value": self.hf_token},
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "WORLD_SIZE", "value": "1"},
                                {"name": "RANK", "value": "0"},
                                {"name": "N_NODES", "value": "1"},
                            ],
                        },
                    }
                ]
            },
        }

        parent = f"projects/{self.project_id}/locations/{self.location}"
        response = self.client.create_custom_job(
            parent=parent, custom_job=custom_job
        )
        job_id = response.name.split("/")[-1]
        return job_id

    def get_training_job_status(self, job_id: str):
        """Gets the status of a Vertex AI Custom Training job."""
        name = self.client.custom_job_path(
            project=self.project_id, location=self.location, custom_job=job_id
        )
        print(name)
        response = self.client.get_custom_job(name=name)

        state = response.state.name
        error_message = None
        if state == "JOB_STATE_FAILED":
            error_message = response.error.message

        return {
            "job_id": job_id,
            "state": state.replace("JOB_STATE_", ""),
            "error": error_message,
        }