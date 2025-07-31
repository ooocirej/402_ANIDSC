from pathlib import Path
from typing import Any, Dict
import torch


class TorchSaveMixin:
    def save(self):
        """save model with torch, all torch models are saved in models folder

        Args:
            suffix (str, optional): suffix of model. Defaults to "".
        """        
        super().save()
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
        # save init args
        torch.save(self._init_args, str(dirpath/"init_args.pth"))
        # save model weights
        torch.save(self.state_dict(), str(dirpath/"model.pth"))

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
        init_args = torch.load(str(dirpath/"init_args.pth"))
        inst = cls(**init_args)
        state_dict = torch.load(str(dirpath/"model.pth"))
        inst.load_state_dict(state_dict)
        return inst
       

    def state_dict(self)->Dict[str, Any]:
        """state dict of model

        Returns:
            Dict[str, Any] : state dictionary
        """        
        state = super().state_dict()
        for i in self.custom_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict:Dict[str, Any]):
        """loads the state dictionary

        Args:
            state_dict (Dict[str, Any]): the state dictionary
        """        
        for i in self.custom_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
        super().load_state_dict(state_dict)