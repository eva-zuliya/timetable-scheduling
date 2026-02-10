from data import (
    DAYS, HOURS_PER_DAY, HORIZON, MAX_SESSION_LENGTH,
    venues, venue_list,
    trainers, eligible,
    courses,
    groups
)

from ortools.sat.python import cp_model
model = cp_model.CpModel()


# ===============================
# SPLIT COURSES INTO SESSIONS
# ===============================
# The purpose of this block is to take each course and break its total duration into smaller sessions,
# where no session is longer than MAX_SESSION_LENGTH hours. This results in a mapping from each course to its list of session indices,
# and a mapping from (course, session index) pairs to the session's actual length (which can be less than MAX_SESSION_LENGTH for the final session).
#
# For example: A course of 6 hours becomes two sessions: [4, 2] if MAX_SESSION_LENGTH is 4.

sessions = {}     # Dictionary: course → list of session indices (e.g., [0, 1, ...])
session_len = {}  # Dictionary: (course, session_index) → session duration (≤MAX_SESSION_LENGTH)

for c, data in courses.items():
    rem = data["dur"]   # Remaining hours of this course
    sessions[c] = []    # Start with empty list of sessions for course c
    k = 0               # Session counter
    while rem > 0:
        l = min(MAX_SESSION_LENGTH, rem)    # Each session takes at most 4 hours, but not more than 'rem'
        sessions[c].append(k)               # Add this session's index to the course's session list
        session_len[(c, k)] = l             # Record session length for this course and session index
        rem -= l                            # Reduce the remaining hours
        k += 1                              # Next session index


# ===============================
# TRAINER ASSIGNMENT (group-subgroup-course)
# ===============================
y = {}
for g in groups:
    for u in groups[g]["subgroups"]:
        for c in groups[g]["courses"]:
            vars_ = []
            for t in trainers:
                if eligible.get((t, c), 0):
                    # y[g, u, c, t] = 1  ⇔ trainer t teaches all sessions of course c
                    # for subgroup u in group g.
                    y[g, u, c, t] = model.NewBoolVar(f"y_{g}_{u}_{c}_{t}")
                    vars_.append(y[g, u, c, t])
            # Exactly one trainer per (group, subgroup, course),
            # shared across all sessions of that course for this subgroup.
            model.Add(sum(vars_) == 1)


# ===============================
# INTERVAL VARIABLES
# ===============================
start, end, day = {}, {}, {}
use, interval = {}, {}

for g in groups:
    for u in groups[g]["subgroups"]:
        for c in groups[g]["courses"]:
            for k in sessions[c]:
                start[g,u,c,k] = model.NewIntVar(0, HORIZON, f"start_{g}_{u}_{c}_{k}")
                end[g,u,c,k]   = model.NewIntVar(0, HORIZON, f"end_{g}_{u}_{c}_{k}")
                model.Add(end[g,u,c,k] == start[g,u,c,k] + session_len[c,k])

                day[g,u,c,k] = model.NewIntVar(0, DAYS-1, f"day_{g}_{u}_{c}_{k}")
                model.AddDivisionEquality(day[g,u,c,k], start[g,u,c,k], HOURS_PER_DAY)

                for v in venue_list:
                    use[g,u,c,k,v] = model.NewBoolVar(f"use_{g}_{u}_{c}_{k}_{v}")
                    interval[g,u,c,k,v] = model.NewOptionalIntervalVar(
                        start[g,u,c,k],
                        session_len[c,k],
                        end[g,u,c,k],
                        use[g,u,c,k,v],
                        f"interval_{g}_{u}_{c}_{k}_{v}"
                    )

                # exactly one venue
                model.Add(sum(use[g,u,c,k,v] for v in venue_list) == 1)


# ===============================
# VENUE CAPACITY (CUMULATIVE)
# ===============================
for v, cap in venues.items():
    model.AddCumulative(
        [interval[g,u,c,k,v]
         for g in groups
         for u in groups[g]["subgroups"]
         for c in groups[g]["courses"]
         for k in sessions[c]],
        [groups[g]["subgroups"][u]
         for g in groups
         for u in groups[g]["subgroups"]
         for c in groups[g]["courses"]
         for k in sessions[c]],
        cap
    )


# ===============================
# TRAINER TIMELINES (NoOverlap)
# ===============================
trainer_interval = {}
for g in groups:
    for u in groups[g]["subgroups"]:
        for c in groups[g]["courses"]:
            for k in sessions[c]:
                for t in trainers:
                    # Only create trainer intervals for (g, u, c, t) combinations
                    # where a trainer assignment variable exists.
                    if (g, u, c, t) in y:
                        trainer_interval[g, u, c, k, t] = model.NewOptionalIntervalVar(
                            start[g, u, c, k],
                            session_len[c, k],
                            end[g, u, c, k],
                            y[g, u, c, t],
                            f"trainer_interval_{g}_{u}_{c}_{k}_{t}",
                        )

# Each trainer's intervals must not overlap in time.
for t in trainers:
    model.AddNoOverlap(
        trainer_interval[g, u, c, k, t]
        for g in groups
        for u in groups[g]["subgroups"]
        for c in groups[g]["courses"]
        for k in sessions[c]
        if (g, u, c, k, t) in trainer_interval
    )


# ===============================
# PREREQUISITES
# ===============================
for g in groups:
    for u in groups[g]["subgroups"]:
        for c in groups[g]["courses"]:
            for prereq in courses[c]["prereq"]:
                for k1 in sessions[prereq]:
                    for k2 in sessions[c]:
                        model.Add(start[g,u,prereq,k1] < start[g,u,c,k2])


# ===============================
# DAILY TRAINEE LIMIT (≤4h)
# ===============================
for g in groups:
    for u in groups[g]["subgroups"]:
        for d in range(DAYS):
            load = []
            for c in groups[g]["courses"]:
                for k in sessions[c]:
                    b = model.NewBoolVar(f"is_{g}_{u}_{c}_{k}_day{d}")
                    model.Add(day[g,u,c,k] == d).OnlyEnforceIf(b)
                    model.Add(day[g,u,c,k] != d).OnlyEnforceIf(b.Not())
                    load.append(b * session_len[c,k])
            model.Add(sum(load) <= MAX_SESSION_LENGTH)


# ===============================
# SYMMETRY BREAKING
# ===============================

# venue order
for g in groups:
    for u in groups[g]["subgroups"]:
        for c in groups[g]["courses"]:
            for k in sessions[c]:
                for i in range(len(venue_list)-1):
                    model.Add(
                        use[g,u,c,k,venue_list[i]] >=
                        use[g,u,c,k,venue_list[i+1]]
                    )

# session order inside course
for g in groups:
    for u in groups[g]["subgroups"]:
        for c in groups[g]["courses"]:
            for k in range(len(sessions[c]) - 1):
                model.Add(start[g,u,c,k] <= start[g,u,c,k+1])


# ===============================
# TRAINER LOAD BALANCING (Objective)
# ===============================
trainer_load = {}
for t in trainers:
    # Upper bound 1000 is safe for small toy instances; can be tightened if desired.
    trainer_load[t] = model.NewIntVar(0, 1000, f"trainer_load_{t}")
    # Sum the durations of all sessions that trainer t teaches.
    load_terms = []
    for g in groups:
        for u in groups[g]["subgroups"]:
            for c in groups[g]["courses"]:
                for k in sessions[c]:
                    if (g, u, c, t) in y:
                        # Each subgroup-course-session that trainer t teaches
                        # contributes its session length to the trainer's load.
                        load_terms.append(session_len[c, k] * y[g, u, c, t])
    model.Add(trainer_load[t] == sum(load_terms))

# Minimize the maximum load across trainers to encourage a balanced assignment.
max_trainer_load = model.NewIntVar(0, 1000, "max_trainer_load")
for t in trainers:
    model.Add(trainer_load[t] <= max_trainer_load)
model.Minimize(max_trainer_load)


# ===============================
# SOLVE
# ===============================
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 20
solver.parameters.num_search_workers = 8

status = solver.Solve(model)
print("Status:", solver.StatusName(status))

import pandas as pd

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    schedule_rows = []
    timetable_cells = {}
    for g, u, c, k in start:
        s = solver.Value(start[g, u, c, k])
        e = solver.Value(end[g, u, c, k])
        start_day = s // HOURS_PER_DAY
        start_hour = s % HOURS_PER_DAY
        end_day = e // HOURS_PER_DAY
        end_hour = e % HOURS_PER_DAY

        # Find assigned venue
        venue_name = None
        venue_occupancy = None
        venue_max_capacity = None
        for v in venue_list:
            if (g, u, c, k, v) in use and solver.Value(use[g, u, c, k, v]):
                venue_name = v
                venue_max_capacity = venues[v]
                venue_occupancy = groups[g]["subgroups"][u]
                break

        # Find assigned trainer
        trainer_name = None
        for t in trainers:
            if (g, u, c, t) in y and solver.Value(y[g, u, c, t]):
                trainer_name = t
                break

        # Get list of trainees names for the subgroup
        trainee_names = groups[g]["subgroups"][u] if "subgroups" in groups[g] and u in groups[g]["subgroups"] else []

        schedule_rows.append({
            "Group": g,
            "Subgroup": u,
            "Trainees": trainee_names,
            "Course": c,
            "Session": k,
            "Start Day": start_day,
            "Start Hour": start_hour,
            "End Day": end_day,
            "End Hour": end_hour,
            "Venue": venue_name,
            "Venue Max Capacity": venue_max_capacity,
            "Venue Occupancy": venue_occupancy,
            "Trainer": trainer_name
        })

        # Fill timetable_cells for simple markdown timetable
        if venue_name is not None and trainer_name is not None:
            key = (venue_name, g, u)
            if key not in timetable_cells:
                timetable_cells[key] = {}
            # Span the session from start 's' up to but not including end 'e'
            for abs_time in range(s, e):
                day = abs_time // HOURS_PER_DAY
                hour = abs_time % HOURS_PER_DAY
                timetable_cells[key][(day, hour)] = trainer_name

    df = pd.DataFrame(schedule_rows)
    print("\nSCHEDULE:")
    print(df)

    # Build improved markdown timetable and write to 'timetable.md'
    timetable_lines = []
    timetable_lines.append("")  # Blank line at start for markdown preview compatibility
    timetable_lines.append("SIMPLE TIMETABLE (Trainer name in slot):")
    # Header
    times = [(d, h) for d in range(DAYS) for h in range(HOURS_PER_DAY)]
    times_header = ["D{}:{:02d}".format(d + 1, h+8) for (d, h) in times]
    first_cols = ["Venue", "Group", "Subgroup"]
    header_row = "| " + " | ".join(first_cols + times_header) + " |"
    sep_row = "|" + "|".join(["---"] * (len(first_cols) + len(times_header))) + "|"
    timetable_lines.append(header_row)
    timetable_lines.append(sep_row)
    # All rows sorted by venue, group, subgroup for stable output
    all_keys_sorted = sorted(timetable_cells.keys())
    for v, g, u in all_keys_sorted:
        cell_map = timetable_cells[(v, g, u)]
        row = []
        for d, h in times:
            trainer = cell_map.get((d, h), '')
            row.append(trainer if trainer else "")
        timetable_lines.append(
            "| {} | {} | {} | {} |".format(v, g, u, " | ".join(row))
        )
    # Write out with proper line endings (just "\n")
    with open("timetable.md", "w", encoding="utf-8") as f:
        for line in timetable_lines:
            f.write(line.rstrip() + "\n")
    print("\nTimetable markdown written to timetable.md")