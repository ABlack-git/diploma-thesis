from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class ImRetConfig:
    gpu: str = MISSING
    data_dir: str = MISSING
    model_dir: str = MISSING
    img_resolution: int = MISSING
