import os
from google.cloud import aiplatform
PROJECT_ID ="<PROJECT_ID>"
REGION = "us-central1"
ENDPOINT_ID="ENDPOINT_ID"

print(f"PROJECT_ID: {PROJECT_ID}")
print(f"REGION: {REGION}")
print(f"ENDPOINT_ID: {ENDPOINT_ID}")

import logging
logging.basicConfig(level=logging.DEBUG)

aiplatform.init(project=PROJECT_ID, location=REGION)

client = aiplatform.gapic.PredictionServiceClient(client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"})

# Define endpoint dictionary (you'll need to adjust based on your deployed endpoints)
endpoints = {
    "vllm_gpu": client.endpoint_path(
        project=PROJECT_ID, location=REGION, endpoint=ENDPOINT_ID
    ) 
}

print(endpoints)
def predict_vllm(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    raw_response: bool,
    lora_weight: str =  "",
):
    # Parameters for inference.
    logging.debug(f"Sending prediction request to endpoint: {endpoints['vllm_gpu']}")
    lora_weight = "<Lora weight path like gs://bucket/path/meta-llama/saves/llama3-8b/lora/sft>"
    # print lora weight path
    print(f"------------- Using lora_weight: {lora_weight} ---------\n")

    
    instance = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "raw_response": raw_response,
    }
    if lora_weight:
        instance["dynamic-lora"] = lora_weight
    instances = [instance]
    # Call the predict method on the correct client object 
    logging.info(f"Sending prediction request: {instances}")
    response = client.predict(
        endpoint=endpoints["vllm_gpu"],  # Pass the endpoint path
        instances=instances,
    )
    logging.debug(f"Received response: {response}")
    for prediction in response.predictions:
        print(prediction)

prompt = input("Enter your prompt: ")
  # @param {type: "string"}
# @markdown If you encounter the issue like `ServiceUnavailable: 503 Took too long to respond when processing`, you can reduce the maximum number of output tokens, such as set `max_tokens` as 20.
max_tokens = 50  # @param {type:"integer"}
temperature = 1.0  # @param {type:"number"}
top_p = 1.0  # @param {type:"number"}
top_k = 1  # @param {type:"integer"}
raw_response = False  # @param {type:"boolean"}


predict_vllm(
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    raw_response=raw_response,
)