import os
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Safe filename for Windows
log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"

log_filepath = os.path.join(log_dir, log_time)

logging.basicConfig(
    level=logging.INFO,
    filename=log_filepath,
    format="[%(asctime)s] - %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

logging.info("Logger initialized successfully")
