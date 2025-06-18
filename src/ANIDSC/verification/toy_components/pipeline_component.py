from abc import ABC, abstractmethod
from typing import get_type_hints, get_args
import inspect
import time
import json
import inspect
from typing import List, Dict, Any, Tuple, get_origin, get_args, Union

class PipelineComponent(ABC):
    @abstractmethod
    def process(self, data):
        pass

    def __or__(self, other:'PipelineComponent')->'Pipeline':
        """attaches pipeline component together
        
        Args:
            other(PipelineComponent): another pipeline component to attach to

        Returns:
            Pipeline: pipeline
        """

        # Get components of self
        if isinstance(self, Pipeline):
            self_components = self.components
        else:
            self_components = [self]
        
        # Get components of other
        if isinstance(other, Pipeline):
            other_components = other.components
        else: 
            other_components = [other]

        # Type check links
        self_last_component, other = self_components[-1], other_components[0]

        # Get return type of last component of self
        raw_return = get_type_hints(self_last_component.process).get('return')
        origin_return = get_origin(raw_return)
        if origin_return is Union:
            # Optional[T]
            union_args = get_args(raw_return)
            not_none = [args for args in union_args if args is not type(None)]
            if len(not_none) == 1:
                self_last_component_return = not_none[0]
            else:
                self_last_component_return = raw_return
        else:
            self_last_component_return = raw_return

        # Get input type of next component, skipping 'self'
        sig = inspect.signature(other.process)
        other_params = [p for p in sig.parameters.values() if p.name != 'self']
        if not other_params:
            raise TypeError(f"{other.__class__.__name__} doesn't have inputs")
        other_input = get_type_hints(other.process).get(other_params[0].name)

        origin_self_return = get_origin(self_last_component_return)
        origin_other_input = get_origin(other_input)

        
        if (origin_self_return is origin_other_input) and (origin_self_return is not None):
            # print("inside origin")
            args_self_return = get_args(self_last_component_return)
            args_other_input = get_args(other_input)
            compatible = True
            for type_returned, type_input in zip(args_self_return, args_other_input):
                if ((type_input is Any) or ((get_origin(type_input) is not None)) and (Any in get_args(type_input))):
                    continue # since it accepts anything
                elif type_returned != type_input:
                    compatible = False
                    break
        else:
            compatible = (self_last_component_return == other_input)

        if not compatible:
            raise TypeError(
                f"Incompatible chaining: {self_last_component.__class__.__name__} returns {self_last_component_return},"
                f"but {other.__class__.__name__} expects {other_input}"
            )
        return Pipeline(self_components + other_components)

    
class Pipeline(PipelineComponent):
    def __init__(self, components: List[PipelineComponent]):
        """ A full pipeline that can be extended with |

        Args:
            components (PipelineComponent): the component of pipeline
        """

        self.components = components
        

    def process(self, data):
        """ sequentially process data over each component
        
        Args:
            data (_type_): the input data
            
        Returns: 
            _type_: output data
        """

        for component in self.components:
            data = component.process(data)

        return data
    

    