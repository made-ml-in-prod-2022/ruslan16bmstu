from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class DownloadParams:
    url: str
    filename: str
    output_folder: str

@dataclass()
class TrainingParams:
    model_type: str = field(default="LogisticRegression")
    random_state: int = field(default=42)

@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)

@dataclass()
class FeatureParams:
    default_features: List[str]
    integer_features: List[str]
    float_features: List[str]
    target_col: Optional[str]

@dataclass()
class ModelParams:
    params: Dict[str, Any]

@dataclass()
class TrainingPipelineParams:
    metrics_path: str
    output_model_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    downloading_params: DownloadParams
    models: ModelParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(cfg: dict) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    return schema.load(cfg)
    #with open(path, "r") as input_stream:
    #    schema = TrainingPipelineParamsSchema()
    #    return schema.load(yaml.safe_load(input_stream))
