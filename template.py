import logging
import os 
from pathlib import Path

logging.basicConfig(level=logging.INFO)

project_name = "mlproject"



list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitor.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
    
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok= True)
        logging.info(f"creating folders {filedir}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath==0)):
        with open(filepath, "w") as f:
            pass
        logging.info(f"creating filepath {filepath}")
        
        
    else:
        logging.info(f"filepath is already exists{filename}")
    
    
