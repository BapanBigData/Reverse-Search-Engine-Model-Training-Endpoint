import logging
import os
from datetime import datetime

# 🔥 Step 1: Create the base logs folder (only 1 folder)
logs_base_folder_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_base_folder_path, exist_ok=True)

# 🔥 Step 2: Create a subfolder inside logs (based on timestamp)
subfolder_name = f"run_on_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"  # Timestamped subfolder name
subfolder_path = os.path.join(logs_base_folder_path, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)  # Ensures logs/run_on_2024_11_21_15_45_23/ exists

# 🔥 Step 3: Create the log file path inside the subfolder (only once per script execution)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(subfolder_path, LOG_FILE)

# 🔥 Step 4: Configure logging (only once for the entire script)
if not logging.getLogger().hasHandlers():  # Prevent multiple logger initializations
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

# 🔥 Get the logger instance (only once)
logger = logging.getLogger(__name__)

# Test log entry
logger.info("Logging has been set up successfully!")

