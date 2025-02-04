import yaml
import os

class ConfigLoader:
    _config = None

    @classmethod
    def load_config(cls, config_path: str = '../config/config.yaml'):
        if cls._config is None:
            with open(config_path, 'r') as file:
                cls._config = yaml.safe_load(file)
        return cls._config
    
    @classmethod
    def get(cls, key, default=None):
        if cls._config is None:
            cls.load_config()
        # Truy cập giá trị bằng key (theo cấp độ)
        keys = key.split(".")
        value = cls._config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default
        
    
