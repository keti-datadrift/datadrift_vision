import logging
import sys
import io
import os

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
# Ensure the console output is UTF-8 encoded
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure the logger globally
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/objdet_textsim_server.log', encoding='utf-8'),  # Ensure log file is in UTF-8
        logging.StreamHandler(sys.stdout)  # Ensure console output uses UTF-8
    ]
)

# Get the logger
logger = logging.getLogger()

