import json
import os
from solver import run_solver
from schema import ModelParams

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    params_file = os.getenv("params_file")

    with open(params_file, 'r') as f:
        params = json.load(f)
        params = ModelParams(**params)

    run_solver(params)
