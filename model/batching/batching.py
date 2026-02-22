from ortools.sat.python import cp_model

# -------------------------
# DATA
# -------------------------

courses = {
    "A": {
        "employees": ["E1","E2","E3","E4"],
        "trainers": 1
    },
    "B": {
        "employees": ["E5","E6","E7","E8","E9"],
        "trainers": 3
    }
}

weeks = [0,1,2,3]  # 4 weeks
shifts = [0,1,2]   # 0=NonShift,1=Shift1,2=Shift2
SHIFT3 = 3

capacity = 4
max_batches_per_course = 5

employee_shift = {
    "E1": {0:1,1:2,2:1,3:2},
    "E2": {0:1,1:2,2:1,3:2},
    "E3": {0:2,1:2,2:1,3:1},
    "E4": {0:3,1:1,2:1,3:2},
    "E5": {0:1,1:1,2:2,3:2},
    "E6": {0:1,1:1,2:2,3:2},
    "E7": {0:2,1:2,2:2,3:1},
    "E8": {0:2,1:2,2:2,3:1},
    "E9": {0:1,1:2,2:1,3:2},
}

# -------------------------
# MODEL
# -------------------------

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

T = model.NewIntVar(0, len(weeks), "Global_Makespan")

# -------------------------
# VARIABLES
# -------------------------

for c in courses:
    employees = courses[c]["employees"]
    M[c] = model.NewIntVar(0, len(weeks), f"M_{c}")

    min_size[c] = model.NewIntVar(0, len(employees), f"min_size_{c}")
    max_size[c] = model.NewIntVar(0, len(employees), f"max_size_{c}")

    for b in range(max_batches_per_course):

        batch_used[(c,b)] = model.NewBoolVar(f"batch_used_{c}_{b}")

        size[(c,b)] = model.NewIntVar(0, capacity, f"size_{c}_{b}")

        for i in employees:
            x[(c,i,b)] = model.NewBoolVar(f"x_{c}_{i}_{b}")

        for w in weeks:
            run[(c,b,w)] = model.NewBoolVar(f"run_{c}_{b}_{w}")
            feasible[(c,b,w)] = model.NewBoolVar(f"feasible_{c}_{b}_{w}")

            for s in shifts:
                z[(c,b,w,s)] = model.NewBoolVar(f"z_{c}_{b}_{w}_{s}")

# -------------------------
# CONSTRAINTS
# -------------------------

for c in courses:

    employees = courses[c]["employees"]
    trainers = courses[c]["trainers"]

    # Each employee assigned exactly once
    for i in employees:
        model.Add(sum(x[(c,i,b)] for b in range(max_batches_per_course)) == 1)

    for b in range(max_batches_per_course):

        # Define batch size
        model.Add(size[(c,b)] == sum(x[(c,i,b)] for i in employees))

        # Capacity
        model.Add(size[(c,b)] <= capacity)

        # Link batch_used
        for i in employees:
            model.Add(x[(c,i,b)] <= batch_used[(c,b)])

        # Batch runs exactly once if used
        model.Add(sum(run[(c,b,w)] for w in weeks) == batch_used[(c,b)])

        # Balance size spread
        model.Add(min_size[c] <= size[(c,b)]).OnlyEnforceIf(batch_used[(c,b)])
        model.Add(max_size[c] >= size[(c,b)]).OnlyEnforceIf(batch_used[(c,b)])

        for w in weeks:

            # At most one shift per week
            model.Add(sum(z[(c,b,w,s)] for s in shifts) <= 1)

            # Run only if feasible
            model.Add(run[(c,b,w)] <= feasible[(c,b,w)])

            # If feasible, must choose exactly one shift
            model.Add(sum(z[(c,b,w,s)] for s in shifts) == 1).OnlyEnforceIf(feasible[(c,b,w)])

            for i in employees:
                s = employee_shift[i][w]

                if s == SHIFT3:
                    # If employee unavailable → batch cannot be feasible
                    model.Add(feasible[(c,b,w)] == 0).OnlyEnforceIf(x[(c,i,b)])
                else:
                    # If feasible and employee assigned → shift must match
                    model.Add(
                        x[(c,i,b)] <= z[(c,b,w,s)]
                    ).OnlyEnforceIf(feasible[(c,b,w)])

        # Course makespan
        for w in weeks:
            model.Add(M[c] >= (w+1) * run[(c,b,w)])

    # Trainer concurrency
    for w in weeks:
        model.Add(
            sum(run[(c,b,w)] for b in range(max_batches_per_course))
            <= trainers
        )

    model.Add(T >= M[c])

# -------------------------
# OBJECTIVE
# -------------------------

BIG = 10000   # Makespan priority
ALPHA = 200   # Fewer batches
GAMMA = 10    # Balance size
BETA = 1      # Flexibility

model.Minimize(
    BIG * T
    + ALPHA * sum(batch_used.values())
    + GAMMA * sum(max_size[c] - min_size[c] for c in courses)
    - BETA * sum(feasible.values())
)

# -------------------------
# SOLVE
# -------------------------

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_search_workers = 8

status = solver.Solve(model)

# -------------------------
# OUTPUT
# -------------------------

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

    print("\nGLOBAL MAKESPAN:", solver.Value(T))
    print()

    for c in courses:
        print("Course", c)
        print(" Makespan:", solver.Value(M[c]))
        print(" Size spread:", solver.Value(max_size[c] - min_size[c]))

        for b in range(max_batches_per_course):
            if solver.Value(batch_used[(c,b)]):

                members = [
                    i for i in courses[c]["employees"]
                    if solver.Value(x[(c,i,b)])
                ]

                run_week = [
                    w+1 for w in weeks
                    if solver.Value(run[(c,b,w)])
                ]

                feasible_weeks = [
                    w+1 for w in weeks
                    if solver.Value(feasible[(c,b,w)])
                ]

                print(" Batch", b)
                print("  Size:", solver.Value(size[(c,b)]))
                print("  Members:", members)
                print("  Runs in week:", run_week)
                print("  Feasible weeks:", feasible_weeks)

        print()

else:
    print("No solution found")