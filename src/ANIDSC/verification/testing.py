import inspect
from typing import get_type_hints, get_args, List, Any, Tuple

from toy_components.feature_extractor import BaseTrafficFeatureExtractor
from toy_components.feature_buffer import BaseFeatureBuffer
from toy_components.detection import BaseDetector
from toy_components.evaluator import BaseEvaluator


class Packet:
    def __init__(self, payload: str, timestamp: float = 0.0):
        self.payload = payload
        self.timestamp = timestamp

def main():
    # get_function_args - 
    # get_function_args()

    # verify_return_types - 
    # processor = test_class.ExampleProcessor()
    # output = processor.process(test_class.Packet("Test"))

    # print("Valid return:", verify_return_types(output))

    # __or__ - 
    extractor = BaseTrafficFeatureExtractor()
    buffer = BaseFeatureBuffer(buffer_size=128)
    detector = BaseDetector(threshold=0.5)
    evaluator = BaseEvaluator(metric_list={})

    #basic chain
    pipeline = extractor | buffer | detector | evaluator

    pkt = Packet("test", timestamp=123.0)
    output = pipeline.process(pkt)
    print("Final output", output)

main()