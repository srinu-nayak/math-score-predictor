import os
import logging
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


log_filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
log_filepath = os.path.join(log_dir, log_filename)


logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format="[%(asctime)s] - %(lineno)d %(name)s - %(levelname)s - %(message)s",
)