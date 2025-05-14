import yaml
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DetectorConfig:
    model_path: str
    confidence: float
    classes: List[str]
    
@dataclass 
class TrackerConfig:
    max_age: int
    n_init: int
    nms_max_overlap: float
    max_cosine_distance: float
    nn_budget: int
    
def load_detector_config(config_path: str) -> DetectorConfig:
    """Load cấu hình detector từ file YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return DetectorConfig(
        model_path=config['model']['path'],
        confidence=config['model']['confidence'],
        classes=config['model']['classes']
    )
    
def load_tracker_config(config_path: str) -> TrackerConfig:
    """Load cấu hình tracker từ file YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return TrackerConfig(
        max_age=config['deep_sort']['max_age'],
        n_init=config['deep_sort']['n_init'],
        nms_max_overlap=config['deep_sort']['nms_max_overlap'],
        max_cosine_distance=config['deep_sort']['max_cosine_distance'],
        nn_budget=config['deep_sort']['nn_budget']
    )