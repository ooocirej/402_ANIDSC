from typing import List, Dict, Any, Tuple
from .pipeline_component import PipelineComponent


class Packet:
    def __init__(self, payload: str, timestamp: float = 0.0):
        self.payload = payload
        self.timestamp = timestamp

        
class BaseTrafficFeatureExtractor(PipelineComponent):
    """
    Toy version of a traffic feature extractor, extending the toy PipelineComponent.
    """
    def __init__(self):
        super().__init__()

    def process(self, packet:Packet, peek=False)->Tuple[List[float], List[Any]]:
        """ toy example, return arbitrary list"""

        feature = [0.1, 0.2, 0.3]
        traffic_vector:List[Any] = [1, "hello", 0.1]
        return feature, traffic_vector
