from abc import abstractmethod
import importlib
import torch


class BaseTorchModel(torch.nn.Module):
    def __init__(self, context, device="cuda", **kwargs):
        torch.nn.Module.__init__(self)
        self.preprocessors = [self.to_tensor, self.to_device]

        self.device = device
        self.optimizer = None
        self.context = context
        
        self.init_model()

    @abstractmethod    
    def init_model(self):
        pass

    def preprocess(self, X):
        """preprocesses the input with preprocessor

        Args:
            X (_type_): input data

        Returns:
            _type_: preprocessed X
        """
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = p(X)
        return X

    def to_device(self, X: torch.Tensor) -> torch.Tensor:
        """preprocessor that converts X to particular device

        Args:
            X (torch.Tensor): the input data

        Returns:
            torch.Tensor: output tensor
        """
        return X.to(self.device)

    def to_tensor(self, X):
        return torch.from_numpy(X).float()

    def to_numpy(self, X: torch.Tensor):
        return X.detach().cpu().numpy()

    @abstractmethod
    def forward(self, X, inference=False):
        pass

    def predict_step(self, X):

        X = self.preprocess(X)

        _, loss = self.forward(X, inference=True)
        
        if loss is None:
            return None

        return loss.detach().cpu().numpy()

    def train_step(self, X):

        X = self.preprocess(X)
        
        if self.optimizer:
            self.optimizer.zero_grad()
        
        _, loss = self.forward(X, inference=False)

        if loss is None:
            return None
        
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().cpu().item()

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self):
        return self.__class__.__name__
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
        
    #     state["state_dict"]={}
    #     for i in self.layers:
    #         state["state_dict"][i] = getattr(self, i).state_dict()
    #         del state["_modules"][i]
        
    #     return state
    
    # def __setstate__(self, state):
    #     # Reconstruct the PyTorch model from state_dict
    #     state_dict=state.pop("state_dict")
    #     self.__dict__.update(state)
    #     for i in self.layers:
    #         getattr(self,i).load_state_dict(state_dict[i])
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the model's full state_dict (includes all parameters/buffers)
        state["model_state_dict"] = self.state_dict()
        return state

    def __setstate__(self, state):
        # Extract the model's state_dict and restore it
        model_state_dict = state.pop("model_state_dict")
        self.__dict__.update(state)
        self.load_state_dict(model_state_dict)
            
