from data import (
    DAYS, HOURS_PER_DAY, HORIZON, MAX_SESSION_LENGTH,
    venues, venue_list,
    trainers, eligible,
    courses,
    groups
)

# print(groups, "\n\n")


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
same_session = {}  # Track if subgroups share the same session

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

# Allow subgroups to share venue session for same course (even across groups)
for c in courses:
    for k in sessions[c]:
        # Collect all (g, u) pairs that take this course
        gu_pairs = [(g, u) for g in groups for u in groups[g]["subgroups"] if c in groups[g]["courses"]]
        
        for i, (g1, u1) in enumerate(gu_pairs):
            for (g2, u2) in gu_pairs[i+1:]:
                same_session[(g1,u1,g2,u2,c,k)] = model.NewBoolVar(f"same_{g1}_{u1}_{g2}_{u2}_{c}_{k}")
                
                # same_session = 1 IFF same start AND same venue AND same trainer
                same_start = model.NewBoolVar(f"same_start_{g1}_{u1}_{g2}_{u2}_{c}_{k}")
                model.Add(start[g1,u1,c,k] == start[g2,u2,c,k]).OnlyEnforceIf(same_start)
                model.Add(start[g1,u1,c,k] != start[g2,u2,c,k]).OnlyEnforceIf(same_start.Not())
                
                same_venue = model.NewBoolVar(f"same_venue_{g1}_{u1}_{g2}_{u2}_{c}_{k}")
                venue_matches = []
                for v in venue_list:
                    both_v = model.NewBoolVar(f"both_{g1}_{u1}_{g2}_{u2}_{c}_{k}_{v}")
                    model.AddMultiplicationEquality(both_v, [use[g1,u1,c,k,v], use[g2,u2,c,k,v]])
                    venue_matches.append(both_v)
                model.Add(sum(venue_matches) == 1).OnlyEnforceIf(same_venue)
                model.Add(sum(venue_matches) == 0).OnlyEnforceIf(same_venue.Not())
                
                same_trainer = model.NewBoolVar(f"same_trainer_{g1}_{u1}_{g2}_{u2}_{c}_{k}")
                trainer_matches = []
                for t in trainers:
                    if (g1,u1,c,t) in y and (g2,u2,c,t) in y:
                        both_t = model.NewBoolVar(f"both_t_{g1}_{u1}_{g2}_{u2}_{c}_{k}_{t}")
                        model.AddMultiplicationEquality(both_t, [y[g1,u1,c,t], y[g2,u2,c,t]])
                        trainer_matches.append(both_t)
                if trainer_matches:
                    model.Add(sum(trainer_matches) >= 1).OnlyEnforceIf(same_trainer)
                    model.Add(sum(trainer_matches) == 0).OnlyEnforceIf(same_trainer.Not())
                else:
                    model.Add(same_trainer == 0)
                
                # same_session = same_start AND same_venue AND same_trainer
                model.AddBoolAnd([same_start, same_venue, same_trainer]).OnlyEnforceIf(same_session[(g1,u1,g2,u2,c,k)])
                model.AddBoolOr([same_start.Not(), same_venue.Not(), same_trainer.Not()]).OnlyEnforceIf(same_session[(g1,u1,g2,u2,c,k)].Not())


# ===============================
# VENUE CAPACITY (CUMULATIVE)
# ===============================
# Capacity constraint now allows multiple subgroups of same course to share venue
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
# TRAINER TIMELINES
# ===============================
# Trainer can teach multiple subgroups simultaneously ONLY if:
# - Same course AND same session AND same venue AND same time
for t in trainers:
    teaching_slots = [(g, u, c, k) for g in groups for u in groups[g]["subgroups"] 
                      for c in groups[g]["courses"] for k in sessions[c] if (g, u, c, t) in y]
    
    for i, (g1, u1, c1, k1) in enumerate(teaching_slots):
        for g2, u2, c2, k2 in teaching_slots[i+1:]:
            # Check if they can share (same course, same session)
            can_share = (c1 == c2 and k1 == k2)
            
            if can_share:
                # They can only overlap if they share the same venue
                both_taught = model.NewBoolVar(f"both_{g1}_{u1}_{g2}_{u2}_{c1}_{k1}_{t}")
                model.AddMultiplicationEquality(both_taught, [y[g1, u1, c1, t], y[g2, u2, c2, t]])
                
                # Check if same venue
                same_venue_vars = []
                for v in venue_list:
                    both_v = model.NewBoolVar(f"both_v_{g1}_{u1}_{g2}_{u2}_{c1}_{k1}_{v}_{t}")
                    model.AddMultiplicationEquality(both_v, [use[g1, u1, c1, k1, v], use[g2, u2, c2, k2, v]])
                    same_venue_vars.append(both_v)
                
                same_venue = model.NewBoolVar(f"same_venue_{g1}_{u1}_{g2}_{u2}_{c1}_{k1}_{t}")
                model.Add(sum(same_venue_vars) >= 1).OnlyEnforceIf(same_venue)
                model.Add(sum(same_venue_vars) == 0).OnlyEnforceIf(same_venue.Not())
                
                # If both taught but NOT same venue, they must not overlap
                must_not_overlap = model.NewBoolVar(f"no_overlap_{g1}_{u1}_{g2}_{u2}_{t}")
                model.AddBoolAnd([both_taught, same_venue.Not()]).OnlyEnforceIf(must_not_overlap)
                model.AddBoolOr([both_taught.Not(), same_venue]).OnlyEnforceIf(must_not_overlap.Not())
                
                order = model.NewBoolVar(f"order_{g1}_{u1}_{g2}_{u2}_{t}")
                model.Add(end[g1, u1, c1, k1] <= start[g2, u2, c2, k2]).OnlyEnforceIf([must_not_overlap, order])
                model.Add(end[g2, u2, c2, k2] <= start[g1, u1, c1, k1]).OnlyEnforceIf([must_not_overlap, order.Not()])
            else:
                # Different course or session - must not overlap
                both_taught = model.NewBoolVar(f"both_{g1}_{u1}_{c1}_{k1}_{g2}_{u2}_{c2}_{k2}_{t}")
                model.AddMultiplicationEquality(both_taught, [y[g1, u1, c1, t], y[g2, u2, c2, t]])
                
                order = model.NewBoolVar(f"order_{g1}_{u1}_{g2}_{u2}_{t}")
                model.Add(end[g1, u1, c1, k1] <= start[g2, u2, c2, k2]).OnlyEnforceIf([both_taught, order])
                model.Add(end[g2, u2, c2, k2] <= start[g1, u1, c1, k1]).OnlyEnforceIf([both_taught, order.Not()])


# ===============================
# PREREQUISITES
# ===============================
# for g in groups:
#     for u in groups[g]["subgroups"]:
#         for c in groups[g]["courses"]:
#             for prereq in courses[c]["prereq"]:
#                 for k1 in sessions[prereq]:
#                     for k2 in sessions[c]:
#                         model.Add(start[g,u,prereq,k1] < start[g,u,c,k2])

for g in groups:
    for u in groups[g]["subgroups"]:
        for c in groups[g]["courses"]:
            for prereq in courses[c]["prereq"]:
                for k1 in sessions.get(prereq, []):
                    for k2 in sessions.get(c, []):
                        key1 = (g, u, prereq, k1)
                        key2 = (g, u, c, k2)

                        if key1 in start and key2 in start:
                            model.Add(start[key1] < start[key2])



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

# session order inside course
for g in groups:
    for u in groups[g]["subgroups"]:
        for c in groups[g]["courses"]:
            for k in range(len(sessions[c]) - 1):
                model.Add(start[g,u,c,k] <= start[g,u,c,k+1])


# ===============================
# OBJECTIVE: MAXIMIZE SHARED SESSIONS
# ===============================
total_shared = model.NewIntVar(0, 10000, "total_shared")
model.Add(total_shared == sum(same_session.values()))

trainer_load = {}
for t in trainers:
    trainer_load[t] = model.NewIntVar(0, 1000, f"trainer_load_{t}")
    load_terms = []
    for g in groups:
        for u in groups[g]["subgroups"]:
            for c in groups[g]["courses"]:
                for k in sessions[c]:
                    if (g, u, c, t) in y:
                        load_terms.append(session_len[c, k] * y[g, u, c, t])
    model.Add(trainer_load[t] == sum(load_terms))

max_trainer_load = model.NewIntVar(0, 1000, "max_trainer_load")
for t in trainers:
    model.Add(trainer_load[t] <= max_trainer_load)

# Maximize shared sessions (primary), minimize trainer load imbalance (secondary)
model.Maximize(total_shared * 1000 - max_trainer_load)


# ===============================
# SOLVE
# ===============================
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 300
solver.parameters.num_search_workers = 8

status = solver.Solve(model)
print("Status:", solver.StatusName(status))
print(f"Objective value: {solver.ObjectiveValue()}")
print(f"Total shared sessions: {solver.Value(total_shared)}")

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
    df.to_csv("schedule.csv", index=False)
    
    # Print sharing information
    print("\n" + "="*60)
    print("SHARED SESSIONS:")
    print("="*60)
    shared_count = 0
    for key, var in same_session.items():
        if solver.Value(var):
            shared_count += 1
            if len(key) == 6:  # (g1, u1, g2, u2, c, k)
                g1, u1, g2, u2, c, k = key
                s1 = solver.Value(start[g1, u1, c, k])
                v1 = [v for v in venue_list if solver.Value(use[g1, u1, c, k, v])][0]
                t1 = [t for t in trainers if (g1, u1, c, t) in y and solver.Value(y[g1, u1, c, t])][0]
                print(f"  {g1}/{u1} + {g2}/{u2} share {c} session {k}")
                print(f"    → Time: {s1}, Venue: {v1}, Trainer: {t1}")
    if shared_count == 0:
        print("  No sessions shared")
    print(f"\nTotal shared: {shared_count}")

    # Build improved markdown timetable and write to 'timetable.md'
    timetable_lines = []
    timetable_lines.append("")  # Blank line at start for markdown preview compatibility
    timetable_lines.append("SIMPLE TIMETABLE (Trainer name in slot):")
    # Header
    times = [(d, h) for d in range(DAYS) for h in range(HOURS_PER_DAY)]
    times_header = ["D{}:{:02d}".format(d + 1, h+8) for (d, h) in times]
    first_cols = ["Venue", "Group", "Subgroup", "Course"]
    header_row = "| " + " | ".join(first_cols + times_header) + " |"
    sep_row = "|" + "|".join(["---"] * (len(first_cols) + len(times_header))) + "|"
    timetable_lines.append(header_row)
    timetable_lines.append(sep_row)
    # All rows sorted by venue, group, subgroup, course for stable output
    # Need to deduce which course is scheduled in each cell (d, h) for this (v, g, u)
    # Build a mapping from (v, g, u, d, h) -> course if possible
    # We'll reconstruct course assignment from schedule_rows
    cell_course_assignment = {}
    for sched in schedule_rows:
        v = sched["Venue"]
        g = sched["Group"]
        u = sched["Subgroup"]
        c = sched["Course"]
        # Get session's absolute start and end (in hours)
        session_idx = sched["Session"]
        s_day, s_hour = sched["Start Day"], sched["Start Hour"]
        e_day, e_hour = sched["End Day"], sched["End Hour"]
        s = s_day * HOURS_PER_DAY + s_hour
        e = e_day * HOURS_PER_DAY + e_hour
        for abs_time in range(s, e):
            d = abs_time // HOURS_PER_DAY
            h = abs_time % HOURS_PER_DAY
            key = (v, g, u, d, h)
            cell_course_assignment[key] = c
    # Collect all unique (venue, group, subgroup, course) combinations that appear in timetable_cells
    all_keys_with_courses = []
    for (v, g, u) in sorted(timetable_cells.keys()):
        # For each (d, h) timeslot look up which courses appear
        cell_map = timetable_cells[(v, g, u)]
        courses_in_this_key = set()
        for (d, h), trainer in cell_map.items():
            c = cell_course_assignment.get((v, g, u, d, h))
            if c is not None:
                courses_in_this_key.add(c)
        # If there are no courses (empty slot), show one row with blank course
        if not courses_in_this_key:
            all_keys_with_courses.append((v, g, u, ""))
        else:
            for c in sorted(courses_in_this_key):
                all_keys_with_courses.append((v, g, u, c))
    # Now, for each (v, g, u, c) row, fill the cells for that course only
    for v, g, u, c in all_keys_with_courses:
        row = []
        for d, h in times:
            # Show trainer only if course matches at this slot
            trainer = timetable_cells[(v, g, u)].get((d, h), "")
            course_here = cell_course_assignment.get((v, g, u, d, h), "")
            cell_value = trainer if course_here == c else ""
            row.append(cell_value)
        timetable_lines.append(
            "| {} | {} | {} | {} | {} |".format(v, g, u, c, " | ".join(row))
        )
    # Write out with proper line endings (just "\n")
    with open("timetable.md", "w", encoding="utf-8") as f:
        for line in timetable_lines:
            f.write(line.rstrip() + "\n")
    print("\nTimetable markdown written to timetable.md")