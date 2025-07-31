from pathlib import Path

class NullSaveMixin:
    def save(self):
        super().save()
        print(f"skipping save for {str(self)}")
        
    def save_state(self, dirpath: Path) -> None:
        return
    
    @classmethod
    def load(cls, path):
        """Load an object from a file using pickle."""
        
        print("didn't load")

    @classmethod
    def load_state(cls, dirpath: Path | None, **attrs):
        return cls(**attrs)