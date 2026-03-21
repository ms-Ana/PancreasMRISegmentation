import os
from dotenv import load_dotenv

load_dotenv()


MODEL_DIR = os.getenv("MODEL_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR")
print(f"MODEL_DIR: {MODEL_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")