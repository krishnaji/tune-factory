
***

# Tune Factory 

Welcome! This guide will walk you through the Tune Factory application. The application uses llama-factory for tuning.

## Getting Started

Before you start, make sure you have the following:

*   An active Google Cloud account.
*   A Google Cloud Storage (GCS) bucket where you'll store your datasets and training outputs.
*   The necessary permissions to access and use the required Google Cloud Vertex AI services.
* An llama-factory docker image 

### Base URL

All API requests will be sent to the following base URL:

```
http://localhost:8000
```

You can replace `localhost` with your server's address if you're running this remotely.

### GCS Bucket

Your datasets, training configurations, and model outputs will be stored in your GCS bucket. Let's call it `shkhose-tune-factory` for now, but feel free to replace this with your actual bucket name.

## Datasets

### Uploading a Dataset

To upload a dataset, you'll send a `POST` request to `/datasets/upload`. This request should include your dataset file in `multipart/form-data` format. Here's how you do it:

```http request
POST http://localhost:8000/datasets/upload
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="alpaca_en_demo.json"
Content-Type: application/json

< ./alpaca_en_demo.json
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

After sending this request, you'll receive a response containing the GCS URL of your uploaded dataset. It'll look something like this:

```json
{
  "gcs_url": "gs://shkhose-tune-factory/datasets/alpaca_en_demo.json"
}
```

This GCS URL will be important for future steps, so keep it handy!

### Listing All Datasets

To see all the datasets you've uploaded, simply send a `GET` request to `/datasets`:

```http request
GET http://localhost:8000/datasets
```

This will return a list of all your datasets.

### Getting a Specific Dataset

If you need to retrieve a specific dataset, you can use its GCS URL. Send a `GET` request to `/datasets/{gcs_url}`, replacing `{gcs_url}` with the actual GCS URL of your dataset:

```http request
GET http://localhost:8000/datasets/gs://shkhose-tune-factory/datasets/alpaca_en_demo.json
```

## Tuning

### Generating a Training Configuration

Before you can start training, you need to generate a training configuration. This is done by sending a `POST` request to `/training/generate_config` with a JSON payload specifying your training parameters.

Here's an example:

```http request
POST http://localhost:8000/training/generate_config
Content-Type: application/json

{
  "dataset_dir": "/gcs/shkhose-tune-factory/datasets",
  "dataset": "alpaca_en_demo",
  "model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
  "output_dir": "/gcs/shkhose-tune-factory/meta-llama/saves/llama3-8b/lora/sft",
  "training_config": {
    "learning_rate": "1.0e-4",
    "template": "llama3",
    "stage": "sft",
    "do_train": "true",
    "finetuning_type": "lora",
    "lora_target": "all",
    "per_device_train_batch_size": "1",
    "gradient_accumulation_steps": "8",
    "num_train_epochs": "1.0",
    "lr_scheduler_type": "cosine",
    "warmup_ratio": "0.1",
    "bf16": "true",
    "ddp_timeout": "180000000",
    "val_size": "0.1",
    "per_device_eval_batch_size": "1",
    "eval_strategy": "steps",
    "eval_steps": "500"
  }
}
```

This will generate a configuration file in your GCS bucket, and you'll receive a response with its GCS URL.

### Listing All Training Configurations

To view all your training configurations, send a `GET` request to `/training/configs`:

```http request
GET http://localhost:8000/training/configs
```

### Starting a Training Job

With your training configuration ready, you can start a training job. Send a `POST` request to `/training/start` with the GCS URL of your configuration file:

```http request
POST http://localhost:8000/training/start
Content-Type: application/json

{
  "config_gcs_url": "gs://shkhose-tune-factory/training_configs/training_config_1178797c.yaml"
}
```

### Checking Training Job Status

To check the status of your training job, you'll need the job ID. Send a `GET` request to `/training/status/{job_id}`, replacing `{job_id}` with your job's ID:

```http request
GET http://localhost:8000/training/status/3604209251972546560
```

## Deployment

### Deploying a Model on vLLM

Once your model is trained, you can deploy it using vLLM. Send a `POST` request to `/deployment/deploy_vllm` with the necessary deployment parameters:

```http request
POST http://localhost:8000/deployment/deploy_vllm
Content-Type: application/json

{
    "model_name": "llama-3-vllm-model-api",
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "service_account": "<123>-compute@developer.gserviceaccount.com",
    "machine_type": "g2-standard-8",
    "accelerator_type": "NVIDIA_L4",
    "accelerator_count": "1",
    "gpu_memory_utilization": "0.9",
    "max_model_len": "4096",
    "dtype": "auto",
    "enable_trust_remote_code": "false",
    "enforce_eager": "false",
    "enable_lora": "true",
    "max_loras": "1",
    "max_cpu_loras": "8",
    "use_dedicated_endpoint": "false",
    "max_num_seqs": "256"
  }
```

You'll receive a response containing the endpoint ID of your deployed model.

### Checking Deployment Status

To check the deployment status, use the endpoint ID you received. Send a `GET` request to `/deployment/status/{endpoint_id}`:

```http request
GET http://localhost:8000/deployment/status/5712755654179946496
```

 ## Prediction

 ### Use Deployed Endpoint

 ```bash
 python predict-vllm.py

 Enter your prompt: what is a car

Prompt:
what is a car?
Output:
 a car is a vehicle that is powered by an internal combustion engine or electric motor and is designed for transporting people or goods. cars are typically designed for use on roads and highways, and are often equipped with features such as air conditioning, radio, and
 ```