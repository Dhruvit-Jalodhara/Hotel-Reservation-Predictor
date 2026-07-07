import logging
import os
from datetime import datetime

# Create a unique log file name using the current date and time.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Path to the logs directory.
logs_path = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't already exist.
os.makedirs(logs_path, exist_ok=True)

# Full path of the log file.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging system.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
