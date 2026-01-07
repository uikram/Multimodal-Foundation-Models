"""
YAML-based Configuration Loader.
Replaces utils/config.py - loads all configs from YAML files into objects.
"""
import yaml
import os
import torch
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Universal config object with attribute access.
    Converts a dictionary (from YAML) into an object (config.attribute).
    """
    
    def __init__(self, config_dict: Dict[str, Any], base_dir: Optional[Path] = None):
        """Initialize config from dictionary."""
        self.base_dir = base_dir or Path(__file__).parent.parent.resolve()
        
        # Store original dict for reliable iteration
        self._original_dict = config_dict
        
        # Recursively convert dictionary to Config objects
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value, self.base_dir))
            else:
                setattr(self, key, value)
        
        # Setup environment
        self._setup_directories()
        self._set_environment_vars()
    
    def _setup_directories(self):
        """Ensure standard directories exist."""
        for attr_name in ['results_dir', 'plots_dir', 'cache_dir', 'output_dir', 'checkpoint_dir']:
            # Use direct dict access to avoid NoneType issues
            path = self.__dict__.get(attr_name)
            
            if path and isinstance(path, (str, Path)):
                path = self._resolve_path(path)
                path.mkdir(parents=True, exist_ok=True)
                setattr(self, attr_name, path)
        
        # Handle nested data directories
        data = self.__dict__.get('data')
        if data and isinstance(data, Config):
            for attr_name in data.__dict__:
                if 'dir' in attr_name and not attr_name.startswith('_'):
                    path = getattr(data, attr_name)
                    if path and isinstance(path, (str, Path)):
                        path = self._resolve_path(path)
                        path.mkdir(parents=True, exist_ok=True)
                        setattr(data, attr_name, path)

    def _resolve_path(self, path):
        """Resolve path relative to project root."""
        path = Path(path)
        if not path.is_absolute():
            path = self.base_dir / path
        return path
    
    def _set_environment_vars(self):
        """Set environment variables for Libraries."""
        # Check top-level then data.cache_dir
        cache_dir = self.__dict__.get('cache_dir')
        if not cache_dir and self.__dict__.get('data'):
            cache_dir = getattr(self.data, 'cache_dir', None)
            
        if cache_dir:
            cache_dir = str(self._resolve_path(cache_dir))
            os.environ['HF_HOME'] = cache_dir
            os.environ['TORCH_HOME'] = cache_dir
            os.environ['XDG_CACHE_HOME'] = cache_dir
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def __getattr__(self, name):
        """Return None for missing attributes."""
        return None
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def load_yaml_config(yaml_path: str) -> Config:
    """Load configuration from YAML file."""
    path = Path(yaml_path)
    base_dir = Path(__file__).parent.parent.resolve()
    
    # Try resolving relative path
    if not path.is_absolute():
        resolved_path = base_dir / path
        if resolved_path.exists():
            path = resolved_path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict, base_dir=base_dir)


def create_model_config(yaml_path: str, verbose: bool = True) -> Config:
    """
    Create model-compatible config from YAML.
    Flattens specific sections and ensures critical keys (model_name) exist.
    """
    cfg = load_yaml_config(yaml_path)
    
    if verbose:
        print(f"Loading config from: {Path(yaml_path).name}")
    
    # Sections to flatten to root (e.g. cfg.model.batch_size -> cfg.batch_size)
    sections = ['model', 'training', 'data', 'evaluation', 'system']
    
    for section_name in sections:
        section = cfg.__dict__.get(section_name)
        if section and isinstance(section, Config):
            # Iterate over the nested config's attributes
            for k, v in section.__dict__.items():
                if not k.startswith('_'):
                    # Copy to root
                    setattr(cfg, k, v)

    # Set defaults if missing
    if cfg.__dict__.get('device') is None:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- ROBUSTNESS FIX ---
    # Ensure cfg.model_name is populated, checking common aliases
    if cfg.__dict__.get('model_name') is None:
        # Check potential aliases in priority order
        for alias in ['model_id', 'name', 'checkpoint_path', 'checkpoint']:
            val = cfg.__dict__.get(alias)
            if val is not None:
                cfg.model_name = val
                break
                
    # Final fallback to avoid "None" errors
    if cfg.__dict__.get('model_name') is None:
         print(f"⚠️ WARNING: 'model_name' not found in {yaml_path}. Defaulting to None.")
            
    return cfg