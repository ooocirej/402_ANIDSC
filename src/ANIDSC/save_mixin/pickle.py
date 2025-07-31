

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
        # stash init args if needed declared in save_attr
        init_args = {
            k: getattr(self, k)
            for k in getattr(self, "save_attr", [])
        }

        with open(dirpath / "init_args.pkl", "wb") as f:
            pickle.dump(init_args, f)

        # now filter __dict__ to only pickle-able items
        state = {}
        for k, v in self.__dict__.items():
            if k in init_args:
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
        """ Load component
            - if no dirpath, instantiate with attrs.
            - If dirpath provided, ignore attrs, use pkl files"""
        
        if dirpath is None:
            return cls(**attrs)
        # load constructor arguments from pickle files
        with open(dirpath / "init_args.pkl", "rb") as f:
            init_args = pickle.load(f)

        #instantiate
        inst = cls(**init_args)

        #restore rest of state
        with open(dirpath / "state.pkl", "rb") as f:
            state = pickle.load(f)
        inst.__dict__.update(state)
        return inst
    