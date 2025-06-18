from typing import Any, Dict, List
from .pipeline_component import PipelineComponent
import time

class BaseEvaluator(PipelineComponent):
    """
    Toy version of BaseEvaluator: processes model output, logs metrics, but does no real I/O.

    Input: Dict[str, Any] (expects keys like 'score', 'threshold', 'batch_num')
    Output: Dict[str, Any] with metrics, 'time', and 'batch_num'.
    """
    def __init__(self,
                 metric_list: List[str],
                 log_to_tensorboard: bool = True,
                 save_results: bool = True):
        super().__init__()
        self.metric_list = metric_list
        # toy flags retained for signature consistency
        self.log_to_tensorboard = log_to_tensorboard
        self.save_results = save_results

    def process(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # Measure time locally to avoid setup dependency
        start_time = time.time()

        result_dict: Dict[str, Any] = {}
        # Log each metric (toy: echo value)
        for name in self.metric_list:
            value = results.get(name)
            result_dict[name] = value
            print(f"[ToyEvaluator] {name} = {value} (batch {results.get('batch_num')})")

        duration = time.time() - start_time
        result_dict['time'] = duration
        result_dict['batch_num'] = results.get('batch_num')
        return result_dict
