from solver import run_solver
import json


if __name__ == "__main__":
    with open('params_all.json', 'r') as f:
        params = json.load(f)

    run_solver(params)
