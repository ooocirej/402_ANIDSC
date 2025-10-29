

from pathlib import Path
import pickle
from pathlib import Path


class PickleSaveMixin:
    def save(self):
        """Save the object to a file using pickle.
        """ 
        super().save()
        save_path = Path(
            self.get_save_path()
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(save_path), 'wb') as file:
            pickle.dump(self, file)
        
    def save_state(self, dirpath: Path) -> None:
        """Save object to file using pickle,
        stashes init args if needed, and dump state.
        """
        dirpath.mkdir(parents=True, exist_ok=True)
        # NEW: Use auto-captured init args if available
        if hasattr(self, '_init_args'):
            init_args = self._init_args.copy()
        else:
            # Fallback: use save_attr for backward compatibility
            init_args = {
                k: getattr(self, k)
                for k in getattr(self, "save_attr", [])
                if hasattr(self, k)
            }

        with open(dirpath / "init_args.pkl", "wb") as f:
            pickle.dump(init_args, f)

        # now filter __dict__ to only pickle-able items
        state = {}
        for k, v in self.__dict__.items():
            if k in init_args or k == '_init_args':  # Skip init args and _init_args itself
                continue
            # try dumping in memory
            try:
                pickle.dumps(v)
            except Exception:
                continue
            state[k] = v

        with open(dirpath / "state.pkl", "wb") as f:
            pickle.dump(state, f)

            
    @classmethod
    def load(cls, path): # dataset_name:str, fe_name:str, file_name:str, name:str, suffix:str=''
        """Load an object from a file using pickle

        Args:
            folder (str): folder of the object
            dataset_name (str): datasetname associated
            fe_name (str): feature extractor name
            file_name (str): file name
            name (str): name of component
            suffix (str, optional): any suffix. Defaults to ''.
        """        
        file_path = Path(path)   
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        
        print(f"Object loaded from {file_path}")
        return obj

    @classmethod
    def load_state(cls, dirpath: Path | None = None, **attrs):
        """Load component."""
        
        if dirpath is None:
            return cls(**attrs)
        
        dirpath = Path(dirpath)
        init_args_file = dirpath / "init_args.pkl"
        state_file = dirpath / "state.pkl"
        
        # Basic validation
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        if not init_args_file.exists():
            raise FileNotFoundError(f"Missing {init_args_file}")
        if not state_file.exists():
            raise FileNotFoundError(f"Missing {state_file}")
        
        # Load
        with open(init_args_file, "rb") as f:
            init_args = pickle.load(f)
        
        inst = cls(**init_args)
        
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        inst.__dict__.update(state)
        
        return inst
    