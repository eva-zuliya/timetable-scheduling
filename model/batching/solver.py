from ortools.sat.python import cp_model
from schema import ModelParams
from .data import read_data
from schema import *

import pandas as pd
pd.set_option("display.max_colwidth", None)


def run_solver(params: ModelParams):
    if params.companies is None or not params.companies:
        return

    WEEKS = [0,1,2,3]  # 4 weeks
    SHIFTS = [0,1,2]   # 0=NonShift,1=Shift1,2=Shift2
    SHIFT3 = 3         # unavailable

    dfs_batch = {}
    for company in params.companies:
        data = read_data(params, company)

        # -------------------------
        # SETS
        # -------------------------
        C = data.courses
        S = {trainee.name: trainee.rotating_shift for trainee in data.shifts.values()}

        # -------------------------
        # CONSTANTS
        # -------------------------
        CAPACITY = next(iter(C.values())).max_venue_capacity_available

        model = cp_model.CpModel()

        x = {}
        run = {}
        z = {}
        batch_used = {}
        M = {}
        feasible = {}
        size = {}
        min_size = {}
        max_size = {}

        T = model.NewIntVar(0, len(WEEKS), "Global_Makespan")

        # -------------------------
        # VARIABLES
        # -------------------------

        for course in C:
            max_batches = C[course].max_batches

            M[course] = model.NewIntVar(0, len(WEEKS), f"M_{course}")

            min_size[course] = model.NewIntVar(0, C[course].count_trainee, f"min_size_{course}")
            max_size[course] = model.NewIntVar(0, C[course].count_trainee, f"max_size_{course}")

            for b in range(max_batches):

                batch_used[(course,b)] = model.NewBoolVar(f"batch_used_{course}_{b}")

                size[(course,b)] = model.NewIntVar(0, CAPACITY, f"size_{course}_{b}")

                for i in C[course].trainees:
                    x[(course,i,b)] = model.NewBoolVar(f"x_{course}_{i}_{b}")

                for w in WEEKS:
                    run[(course,b,w)] = model.NewBoolVar(f"run_{course}_{b}_{w}")
                    feasible[(course,b,w)] = model.NewBoolVar(f"feasible_{course}_{b}_{w}")

                    for s in SHIFTS:
                        z[(course,b,w,s)] = model.NewBoolVar(f"z_{course}_{b}_{w}_{s}")

        # -------------------------
        # CONSTRAINTS
        # -------------------------

        for course in C:
            trainees = C[course].trainees
            trainers = C[course].count_trainers
            max_batches = C[course].max_batches

            # Each employee assigned exactly once
            for i in trainees:
                model.Add(sum(x[(course,i,b)] for b in range(max_batches)) == 1)

            for b in range(max_batches):

                # Define batch size
                model.Add(size[(course,b)] == sum(x[(course,i,b)] for i in trainees))

                # Capacity
                model.Add(size[(course,b)] <= CAPACITY)

                # Link batch_used
                for i in trainees:
                    model.Add(x[(course,i,b)] <= batch_used[(course,b)])

                # Batch runs exactly once if used
                model.Add(sum(run[(course,b,w)] for w in WEEKS) == batch_used[(course,b)])

                # Balance size spread
                model.Add(min_size[course] <= size[(course,b)]).OnlyEnforceIf(batch_used[(course,b)])
                model.Add(max_size[course] >= size[(course,b)]).OnlyEnforceIf(batch_used[(course,b)])

                for w in WEEKS:

                    # At most one shift per week
                    model.Add(sum(z[(course,b,w,s)] for s in SHIFTS) <= 1)

                    # Run only if feasible
                    model.Add(run[(course,b,w)] <= feasible[(course,b,w)])

                    # If feasible, must choose exactly one shift
                    model.Add(
                        sum(z[(course,b,w,s)] for s in SHIFTS) == 1
                    ).OnlyEnforceIf(feasible[(course,b,w)])

                    for i in trainees:
                        if i in S and w < len(S[i]):
                            if s == SHIFT3:
                                model.Add(feasible[(course,b,w)] == 0).OnlyEnforceIf(x[(course,i,b)])
                            else:
                                model.Add(
                                    x[(course,i,b)] <= z[(course,b,w,s)]
                                ).OnlyEnforceIf(feasible[(course,b,w)])

                # Course makespan
                for w in WEEKS:
                    model.Add(M[course] >= (w+1) * run[(course,b,w)])

            # # Trainer concurrency
            # for w in WEEKS:
            #     model.Add(
            #         sum(run[(course,b,w)] for b in range(max_batches))
            #         <= trainers
            #     )

            model.Add(T >= M[course])

        # -------------------------
        # OBJECTIVE
        # -------------------------

        BIG = 10000   # Makespan priority
        ALPHA = 200   # Fewer batches
        GAMMA = 10    # Balance size
        BETA = 1      # Flexibility reward

        model.Minimize(
            BIG * T
            + ALPHA * sum(batch_used.values())
            + GAMMA * sum(max_size[course] - min_size[course] for course in C)
            - BETA * sum(feasible.values())
        )

        # -------------------------
        # SOLVE
        # -------------------------

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = params.max_time_in_seconds
        solver.parameters.num_search_workers = params.num_search_workers

        status = solver.Solve(model)

        # -------------------------
        # OUTPUT
        # -------------------------

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("\nGLOBAL MAKESPAN:", solver.Value(T))

            rows = []
            for course in C:
                max_batches = C[course].max_batches
                batch_counter = 1

                for b in range(max_batches):
                    if solver.Value(batch_used[(course, b)]):

                        members = [
                            i for i in C[course].trainees
                            if solver.Value(x[(course, i, b)])
                        ]

                        # Determine overlapped shift per week
                        week_shifts = {}

                        for w in WEEKS:

                            # Default = unavailable
                            shift_value = SHIFT3

                            if solver.Value(feasible[(course, b, w)]):
                                for s in SHIFTS:
                                    if solver.Value(z[(course, b, w, s)]):
                                        shift_value = s
                                        break

                            week_shifts[w] = shift_value

                        for trainee in members:
                            rows.append({
                                "company": company,
                                "course_name": course,
                                "batch_no": batch_counter,
                                "trainee_id": trainee,
                                "week1": week_shifts[0],
                                "week2": week_shifts[1],
                                "week3": week_shifts[2],
                                "week4": week_shifts[3],
                            })

                        batch_counter += 1

            df = pd.DataFrame(rows)
            if not df.empty:
                # Get all columns except "trainee_id"
                group_cols = [col for col in df.columns if col != "trainee_id"]
                # For each unique combination in the grouping columns, count the number of trainees
                pivot = (df
                         .groupby(group_cols, as_index=False)
                         .agg(trainee_count=('trainee_id', 'count')))
                print("\nPivot table (with all columns except 'trainee_id', + count of trainees):\n", pivot)
            
            # print(df)
            dfs_batch[company] = df.copy()

        else:
            print(f"No batch solution for {company} found")


    if len(dfs_batch):
        batch = pd.concat(dfs_batch.values(), ignore_index=True)
        batch.to_csv(f"export/{params.report_name}_batch.csv", index=False)

        print(f"Batch report has been exported to export/{params.report_name}_batch.csv")
    else:
        print("No solution found")