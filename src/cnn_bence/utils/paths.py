
from pathlib import Path

def get_project_root():
    current_file = Path(__file__)
    for parent in current_file.parents:
        if (parent/"pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find root file pyproject.toml")

def get_data_dir():
    return get_project_root()/"data" 
        