from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class TrainingConfig(BaseModel):
    learning_rate: float = Field(..., example=0.001)   
    template: str = Field(..., example="llama3")
    stage: str = Field(..., example="sft")
    do_train: bool = Field(..., example=True)
    finetuning_type: str = Field(..., example="lora")
    lora_target: str = Field(..., example="all")
    per_device_train_batch_size: int = Field(..., example=1)
    gradient_accumulation_steps: int = Field(..., example=8)
    num_train_epochs: float = Field(..., example=3.0)
    lr_scheduler_type: str = Field(..., example="cosine")
    warmup_ratio: float = Field(..., example=0.1)
    bf16: bool = Field(..., example=True)
    ddp_timeout: int = Field(..., example=180000000)
    val_size: float = Field(..., example=0.1)
    per_device_eval_batch_size: int = Field(..., example=1)
    eval_strategy: str = Field(..., example="steps")
    eval_steps: int = Field(..., example=500)



class GenerateConfigSchema(BaseModel):
    dataset_dir: str = Field(..., example="datasets/my_dataset.csv")
    model_name_or_path: str = Field(..., example="meta-llama/Meta-Llama-3-8B-Instruct")
    output_dir: str = Field(..., example="meta-llama/saves/llama3-8b/lora/sft")
    dataset: str = Field(..., example="my_dataset")
    training_config: TrainingConfig

class StartTrainingSchema(BaseModel):
    config_gcs_url: str = Field(..., example="gs://your-gcs-bucket/training_configs/training_config_123.yaml")

class DeployModelSchema(BaseModel):
    model_id: str = Field(..., description="ID of the trained model in Vertex AI Model Registry")

class DatasetItem(BaseModel):
    filepath: str = Field(..., example="datasets/my_dataset.csv")
    gcs_url: str = Field(..., example="gs://your-gcs-bucket/datasets/my_dataset.csv")

class TrainingConfigItem(BaseModel):
    filepath: str = Field(..., example="training_configs/training_config_123.yaml")
    gcs_url: str = Field(..., example="gs://your-gcs-bucket/training_configs/training_config_123.yaml")

class TrainingJobStatus(BaseModel):
    job_id: str
    state: str
    error: Optional[str] = None

class DeploymentJobStatus(BaseModel):
    endpoint_id: str
    state: str
    error: Optional[str] = None


class VLLMDeployModelSchema(BaseModel):
    model_name: str = Field(..., example="my-vllm-model")
    model_id: str = Field(..., example="meta-llama/Meta-Llama-3-8B-Instruct")
    service_account: str = Field(..., example="your-service-account@your-project.iam.gserviceaccount.com")
    machine_type: str = Field("g2-standard-8", example="g2-standard-8")
    accelerator_type: str = Field("NVIDIA_L4", example="NVIDIA_L4")
    accelerator_count: int = Field(1, example=1)
    gpu_memory_utilization: float = Field(0.9, example=0.9)
    max_model_len: int = Field(4096, example=4096)
    dtype: str = Field("auto", example="auto")
    enable_trust_remote_code: bool = Field(False, example=False)
    enforce_eager: bool = Field(False, example=False)
    enable_lora: bool = Field(False, example=False)
    max_loras: int = Field(1, example=1)
    max_cpu_loras: int = Field(8, example=8)
    use_dedicated_endpoint: bool = Field(False, example=False)
    max_num_seqs: int = Field(256, example=256)
    model_type: str = Field(None, example=None)