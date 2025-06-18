from typing import List, Any, Tuple, Union
from .pipeline_component import PipelineComponent
import numpy as np


class BaseFeatureBuffer(PipelineComponent):
    """
    Toy version of BaseFeatureBuffer: satisfies the signature without real I/O or buffering.

    Input: Tuple[List[Any], List[Any]]
    Output: Tuple[List[Any], List[Any]] (always passes through)
    """
    def __init__(self, buffer_size: int = 256, save_features: bool = True, save_meta: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.save_features = save_features
        self.save_meta = save_meta

    def setup(self):
        super().setup()
        # Toy: record buffer configuration in context
        self.context['buffer_size'] = self.buffer_size

    def process(self, data: Tuple[List[Any], List[Any]]) -> Union[None, Tuple[List[Any]]]:
        # No real buffering; simply forward data
        return data

    def teardown(self):
        # Toy: no resources to close
        super().teardown()
