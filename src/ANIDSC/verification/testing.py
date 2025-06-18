import test_class
import inspect
from typing import get_type_hints, get_args, List, Any, Tuple

from toy_components.feature_extractor import BaseTrafficFeatureExtractor
from toy_components.feature_buffer import BaseFeatureBuffer
from toy_components.detection import BaseDetector
from toy_components.evaluator import BaseEvaluator

def get_function_args(): 
    # packet = test_class.Packet("Hi")
    # processor = test_class.ExampleProcessor()

    type_hints = get_type_hints(test_class.ExampleProcessor.process) # 
    signature = inspect.signature(test_class.ExampleProcessor.process)  # get function details
    # print(signature)
    # print(type_hints)

    for name, param in signature.parameters.items(): #Extract parameter names and types 
        param_type = type_hints.get(name, "No annotation")
        print(f"{name}: {param_type}")

    return_type = get_type_hints(test_class.ExampleProcessor.process).get('return') # Get return type
    return_types = get_args(return_type)

    for i, type in enumerate(return_types):
        print(f"Return element {i}: {type}")    

def verify_return_types(output: Tuple[List[float], List[Any]])->bool:
    if not isinstance(output, tuple) or len(output) != 2:
        return False
    
    list1, list2 = output

    # check if first item is a list of floats
    if not isinstance(list1, list) or not all(isinstance(x, float) for x in list1):
        return False
    
    # check if second item is a list
    if not isinstance(list2, list):
        return False
    
    return True

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