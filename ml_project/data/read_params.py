from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from marshmallow_dataclass import class_schema


@dataclass()
class DownloadParams:
    url: str
    file_path: str


@dataclass()
class TrainingParams:
    model_type: str
    params: Dict[Any, Any]
    random_state: int


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
class TrainingPipelineParams:
    metrics_path: str
    output_model_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    model: TrainingParams
    downloading_params: DownloadParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(cfg: dict) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    return schema.load(cfg)
