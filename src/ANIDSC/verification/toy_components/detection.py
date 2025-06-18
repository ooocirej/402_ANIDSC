from typing import Any, Dict, List, Tuple
from .pipeline_component import PipelineComponent

class BaseDetector(PipelineComponent):
    """
    Toy detection component simulating an online outlier detector.

    Input: Tuple[List[float], Any]
    Output: Dict[str, Any] with keys 'score', 'threshold', and 'batch_num'.
    """
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.batch_num = 0

    def process(self, input_data: Tuple[List[float], Any]) -> Dict[str, Any]:
        features, metadata = input_data
        # Compute a toy score: mean of features
        if features:
            score = sum(features) / len(features)
        else:
            score = 0.0
        self.batch_num += 1
        return {
            "score": score,
            "threshold": self.threshold,
            "batch_num": self.batch_num
        }