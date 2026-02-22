import json
import os
# from solver import run_solver
from schema import ModelParams
from model.batching.solver import run_solver as batching_solver
from model.scheduling.solver import run_solver as scheduling_solver

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    params_file = os.getenv("params_file")

    with open(params_file, 'r') as f:
        params = json.load(f)
        params = ModelParams(**params)

    if params.is_splitting_batch:
        params.file_master_course_batch = [f"export/{params.report_name}_batch.csv"]
        batching_solver(params)

    if params.is_scheduling_course:
        scheduling_solver(params)

    
