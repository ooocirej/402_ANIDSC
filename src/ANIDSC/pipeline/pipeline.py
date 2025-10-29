
import importlib
import time
from typing import Dict, Optional
from pathlib import Path

from tqdm import tqdm
import yaml
from ..component.pipeline_component import PipelineComponent
from ..save_mixin.yaml import YamlSaveMixin


class Pipeline(YamlSaveMixin, PipelineComponent):
    
    def __init__(self, **kwargs):
        """A full pipeline that can be extended with |

        Args:
            components (PipelineComponent): the component of pipeline
        """        
        super().__init__(component_type="pipeline")
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.save_attr.append(key)
        self.start_time=None
        self.prefix=[]

    def start(self):
        
        pbar = tqdm()
        
        while True:
            comp_type=self.process()
            
            pbar.update(1)
            if comp_type=="data_source":
                break
            
        ds     = self.manifest["data_source"]["attrs"]["dataset_name"]
        fe     = self.perform_action("feature_extractor", "__str__") or \
                 self.manifest["data_source"]["attrs"]["fe_name"]
        fname  = self.manifest["data_source"]["attrs"]["file_name"]

        # NEW: Include pipeline type to avoid conflicts
        pipeline_type = self.get_pipeline_type()
        save_dir = Path(ds) / fe / fname / pipeline_type
        save_dir.mkdir(parents=True, exist_ok=True)

        return self.save_state() # return Path used

    def get_pipeline_type(self) -> str:
        """get pipeline type from component structure."""
        components = set(self.components.keys())
    
        if "feature_extractor" in components and "feature_buffer" in components:
            return "feature_extraction"
        
        if "model" in components and "evaluator" in components:
            return "detection"
        
        if "pipeline" in components:
            return "graph_detection"
        
        return "custom"
    
    def save_metadata(self, base_dir: Path):
        """Save minimal metadata for downstream pipelines."""
        if "feature_extractor" not in self.components:
            return
        
        fe = self.components["feature_extractor"]
        metadata = {
            "fe_attrs": {
                k: getattr(fe, k) 
                for k in getattr(fe, "save_attr", [])
            }
        }
        
        with open(base_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)
    
    def save_state(self, dirpath: Path = None) -> Path:
        """Save pipeline state."""
        
        if dirpath is None:
            ds = self.manifest["data_source"]["attrs"]["dataset_name"]
            fe = (
                self.perform_action("feature_extractor", "__str__")
                or self.manifest["data_source"]["attrs"]["fe_name"]
            )
            fname = Path(self.manifest["data_source"]["attrs"]["file_name"]).stem
            pipeline_type = self.get_pipeline_type()
            root = Path(ds) / fe / fname / pipeline_type
        else:
            root = Path(dirpath)

        root.mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest = {}
        for name, comp in self.components.items():
            attrs = {k: getattr(comp, k) for k in getattr(comp, "save_attr", [])}
            manifest[name] = {
                "module": comp.__class__.__module__,
                "class": comp.__class__.__name__,
                "attrs": attrs,
                "file": name,
            }
        
        with open(root / "pipeline_config.yaml", "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, indent=2)

        # Save each component
        for name, comp in self.components.items():
            comp_dir = root / name
            comp.save_state(comp_dir)

        return root
    
    @classmethod
    def load_state(cls, dirpath: Path) -> "Pipeline":
        """Load Pipeline from directory."""
        
        root = Path(dirpath)
        
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint not found: {root}")
        
        manifest_file = root / "pipeline_config.yaml"
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Missing pipeline_config.yaml in {root}. "
                f"Not a valid checkpoint."
            )

        manifest = yaml.safe_load(open(manifest_file))

        pipeline = cls()
        pipeline._root = root
        pipeline.manifest = manifest
        pipeline.components = pipeline.load_components(manifest)
        
        for comp in pipeline.components.values():
            comp.on_load()

        return pipeline
    
    # def load_components(self, manifest):
    #     components = {}        
    #     for type, meta in manifest.items():
    #         module = importlib.import_module(meta["module"])
    #         component_cls = getattr(module, meta["class"])
    #         if meta.get("file", False):
    #             file_path = meta["file"]
    #             comp = component_cls.load(file_path)
                
    #         else:
    #             comp = component_cls(**meta.get("attrs", {}))
    #         comp.parent_pipeline=self
    #         components[type] = comp
    #     return components

    def load_components(self, manifest):
        components = {}        
        # for type, meta in manifest.items():
        #     module = importlib.import_module(f"ANIDSC.{type}")
        #     component_cls = getattr(module, meta["class"])
        #     if meta.get("file", False):
        #         file_path = meta["file"]
        #         comp = component_cls.load(file_path)
                
        #     else:
        #         comp = component_cls(**meta.get("attrs", {}))
        #     comp.parent_pipeline=self
        #     components[type] = comp

        for name, meta in manifest.items():
            module_path = meta.get("module", f"ANIDSC.{name}")
            module = importlib.import_module(module_path)
            component_cls = getattr(module, meta["class"])
            attrs = meta.get("attrs", {})
            file_path = meta.get("file")
            state_dir = (self._root / Path(file_path)) if file_path else None
            comp = component_cls.load_state(state_dir, **attrs)

            comp.parent_pipeline = self
            components[name] = comp

        return components
        
    def on_load(self):
        self.components=self.load_components(self.manifest)
        for key, comp in self.components.items():
            comp.on_load()
    
    def add_prefix(self, prefix):
        self.prefix.append(prefix)
    
    def setup(self):
        
        self.components=self.load_components(self.manifest)
        
        for key, comp in self.components.items():
            comp.setup()
            

    def perform_action(self, comp_type, action):
        if comp_type in self.components:
            return getattr(self.components[comp_type], action)()
    
    
    def process(self, data=None):
        """sequentially process data over each component

        Args:
            data (_type_): the input data

        Returns:
            _type_: output data
        """
        self.start_time=time.time()
        for comp_type, component in self.components.items():
            data = component.preprocess(data)
            data = component.process(data)
            data = component.postprocess(data)
            if data is None: 
                break
        return comp_type
            
            
    # def get_save_path_template(self):
    #     fe_name=self.perform_action('feature_extractor', '__str__')
    #     if fe_name is None:
    #         fe_name=self.get_attr("data_source","fe_name")
        
    #     if len(self.prefix)==0:
        
    #         return f"{self.get_attr('data_source','dataset_name')}/{fe_name}/saved_components/{{}}/{self.get_attr('data_source','file_name')}/{{}}.{{}}"

    #     else:
    #         prefix_str="/".join(self.prefix)
    #         return f"{self.get_attr('data_source','dataset_name')}/{fe_name}/saved_components/{{}}/{self.get_attr('data_source','file_name')}/{prefix_str}/{{}}.{{}}"

    def get_save_path_template(self) -> str:
        """ Returns a format string with two placeholders:
            1) component name
            2) filename (e.g. 'init_args' or 'state')

        Example result:
            'test_data/AfterImage/benign_lenovo_bulb/{component}/{file}.pkl'
        """
        # dataset
        ds = self.manifest["data_source"]["attrs"]["dataset_name"]
        # feature-extractor name
        fe = (
            self.perform_action("feature_extractor", "__str__")
            or self.manifest["data_source"]["attrs"]["fe_name"]
        )
        #file name (pcap basename)
        fname = Path(self.manifest["data_source"]["attrs"]["file_name"]).stem

        # build the root: e.g. "test_data/AfterImage/benign_lenovo_bulb"
        root = Path(ds) / fe / fname

        # return the template under that root:
        #   {component} → e.g. "data_source" or "feature_extractor"
        #   {file}      → e.g. "init_args" or "state"
        return str(root / "{component}" / "{file}.pkl")


    def get_attr(self, comp_type, attr, default=None):
        if comp_type in self.components:
            return getattr(self.components[comp_type], attr, default)
        else:
            return self.request_attr(comp_type, attr, default)
        
    def __eq__(self, other: 'Pipeline'):
        same_class=self.__class__==other.__class__ 
        if not same_class:
            return False
        
       
        return self.components==other.components

    def __str__(self):
        if len(self.prefix)==0:
            return "->".join([str(component) for _, component in self.components.items()])
        else:
            return "/".join(self.prefix) +"/"+ "->".join([str(component) for _, component in self.components.items()])
    
