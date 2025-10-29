from pathlib import Path
from typing import Any, Dict
import warnings
import torch


class TorchSaveMixin:
    def save(self):
        """save model with torch, all torch models are saved in models folder

        Args:
            suffix (str, optional): suffix of model. Defaults to "".
        """        
        # checkpoint = {
        #     "model_state_dict": self.state_dict(),
        # }
        # if hasattr(self, "optimizer"):
        #     checkpoint["optimizer_state_dict"] = (self.optimizer.state_dict(),)
        ckpt_path = Path(self.get_save_path())
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, str(ckpt_path))

    def save_state(self, dirpath: Path) -> None:

        dirpath.mkdir(parents=True, exist_ok=True)
        
        # Check if _init_args exists
        if not hasattr(self, '_init_args'):
            warnings.warn(
                f"{self.__class__.__name__}: _init_args not set. "
                f"Model may not load correctly. "
                f"Ensure your __init__ calls super().__init__() or sets _init_args."
            )
            # Try to extract from save_attr as fallback
            self._init_args = {
                k: getattr(self, k)
                for k in getattr(self, "save_attr", [])
            }
        
        # Save init args
        torch.save(self._init_args, str(dirpath / "init_args.pth"))
        
        # Save model weights
        torch.save(self.state_dict(), str(dirpath / "model.pth"))
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(self, str(dirpath / "full_model.pt"))

    @classmethod
    def load(cls, path):
        """loads the parameters of torch model

        Args:
            suffix (str, optional): optional suffix. Defaults to "".
        """        
        ckpt_path = Path(path)   
        if not ckpt_path.exists():
            return None
        model = torch.load(ckpt_path)

        # model = cls()
        # model.setup()
        # model.load_state_dict(checkpoint["model_state_dict"])
        # if "optimizer_state_dict" in checkpoint.keys():
        #     model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"][0])
        return model 
    
    @classmethod
    def load_state(cls, dirpath: Path):
        """Load torch model."""
        
        dirpath = Path(dirpath)
        init_args_file = dirpath / "init_args.pth"
        model_file = dirpath / "model.pth"
        
        # Basic validation
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        if not init_args_file.exists():
            raise FileNotFoundError(f"Missing {init_args_file}")
        if not model_file.exists():
            raise FileNotFoundError(f"Missing {model_file}")
        
        # Load
        init_args = torch.load(str(init_args_file))
        inst = cls(**init_args)
        state_dict = torch.load(str(model_file))
        inst.load_state_dict(state_dict)
        
        return inst
       

    def state_dict(self)->Dict[str, Any]:
        """state dict of model

        Returns:
            Dict[str, Any] : state dictionary
        """        
        state = super().state_dict()
        for i in getattr(self, "custom_params", []): # no-op when no custom_params
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict:Dict[str, Any]):
        """loads the state dictionary

        Args:
            state_dict (Dict[str, Any]): the state dictionary
        """        
        for i in getattr(self, "custom_params", []): # no-op when no custom_params
            setattr(self, i, state_dict[i])
            del state_dict[i]
        super().load_state_dict(state_dict)

    def on_load(self):
        self.to(getattr(self, "device", "cpu"))
        try: self.eval()
        except Exception: pass