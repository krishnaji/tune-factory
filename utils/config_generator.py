import yaml
from schemas import TrainingConfig

def generate_training_config(dataset_dir: str, model_name_or_path: str,  output_dir: str,dataset:str, training_config: TrainingConfig) -> str:
    """Generates a training configuration YAML for llama-factory."""
    config = {
        "dataset_dir": dataset_dir,
        "model_name_or_path": model_name_or_path,
         "output_dir": output_dir,
         "dataset" : dataset,
        **training_config.model_dump(),  # Use model_dump() to unpack the training config
    }
    return yaml.dump(config)