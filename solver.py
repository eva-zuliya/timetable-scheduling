import pandas as pd
from ortools.sat.python import cp_model
from schema import *
from utils import hour_index_to_time
import datetime
from data import read_data


def run_solver(params: ModelParams):
    data = read_data(params)

    # DAYS = data['days']
    # HOURS_PER_DAY = data['hours_per_day']
    # HORIZON = data['horizon']
    # MAX_SESSION_LENGTH = data['max_session_length']
    # venues = data['venues']
    # venue_list = data['venue_list']
    # virtual_venue_list = data['virtual_venue_list']
    # trainers = data['trainers']
    # eligible = data['eligible']
    # courses = data['courses']
    # groups = data['groups']
    # groups_trainee = data['groups_trainee']
    # calendar = data['calendar']
    # weekend_list = data['weekend_list']

    # is_considering_shift = data['is_considering_shift']
    # is_using_global_sequence = data['is_using_global_sequence']


    model = cp_model.CpModel()

    # ===============================
    # SETS
    # ===============================
    G = data.groups
    T = data.trainers
    V = data.venues
    C = data.courses
    D = params.days

    # ===============================
    # CONSTANTS
    # ===============================
    DAYS = params.days
    HOURS_PER_DAY = params.hours_per_day
    HORIZON = DAYS * HOURS_PER_DAY
    CALENDAR = data.calendar
    MAX_SESSION_LENGTH = params.maximum_session_length


    # ===============================
    # SESSION INDEX
    # ===============================
    # For each course, make a session for every subgroup in every group that takes that course.
    S = {}
    for course in C:
        max_num_sessions = 0
        for group in G:
            if course in G[group].courses:
                max_num_sessions += len(G[group].trainees)  # Each subgroup in this group needs a session for this course

        if max_num_sessions>0:
            S[course] = [0]

        # S could look like: {'C1': [0, 1, 2], 'C2': [0, 1]}
        # Where S['C1'] = [0,1,2] means there are 3 sessions for course 'C1'


    # ===============================
    # SESSION VARIABLES
    # ===============================
    active_session = {}
    start_session = {}
    end_session = {}
    day_session = {}
    venue_session = {}
    trainer_session = {}

    for course in C:
        if course in S:
            dur = C[course].duration

            for session in S[course]:
                active_session[course, session] = model.NewBoolVar(f"active_{course}_{session}")

                start_session[course, session] = model.NewIntVar(0, HORIZON, f"start_{course}_{session}")
                end_session[course, session] = model.NewIntVar(0, HORIZON, f"end_{course}_{session}")

                model.Add(
                    end_session[course, session] == start_session[course, session] + dur
                )

                day_session[course, session] = model.NewIntVar(0, DAYS - 1, f"day_{course}_{session}")
                model.AddDivisionEquality(
                    day_session[course, session], start_session[course, session], HOURS_PER_DAY
                )

                # Same-day constraint
                end_day = model.NewIntVar(0, DAYS - 1, f"endday_{course}_{session}")
                model.AddDivisionEquality(
                    end_day, end_session[course, session] - 1, HOURS_PER_DAY
                )
                
                model.Add(
                    day_session[course, session] == end_day
                ).OnlyEnforceIf(active_session[course, session])

                # Venue Assignment
                for venue in V:
                    venue_session[course, session, venue] = model.NewBoolVar(f"venue_{course}_{session}_{venue}")

                model.Add(
                    sum(
                        venue_session[course, session, venue]
                            for venue in V
                    ) == active_session[course, session]
                )

                # Trainer Assignment
                eligible = {
                    (trainer, course): 1
                    for trainer in T for course in T[trainer].eligible
                }

                for trainer in T:
                    if eligible.get((trainer, course), 0):
                        trainer_session[course, session, trainer] = model.NewBoolVar(f"trainer_{course}_{session}_{trainer}")

                model.Add(
                    sum(
                        trainer_session[course, session, trainer]
                        for trainer in T
                            if (course, session, trainer) in trainer_session
                    ) == active_session[course, session]
                )


    # ===============================
    # SUBGROUP → SESSION ASSIGNMENT
    # ===============================
    assign = {}

    for group in G:
        for course in G[group].courses:
            assign_vars = []

            for session in S[course]:
                assign[group, course, session] = model.NewBoolVar(f"assign_{group}_{course}_{session}")
                assign_vars.append(assign[group, course, session])

            model.Add(sum(assign_vars) == 1)


    # ===============================
    # SESSION ACTIVE IF USED
    # ===============================
    for course in C:
        if course in S:
            for session in S[course]:

                model.AddMaxEquality(
                    active_session[course, session],
                    [
                        assign[group, course, session]
                            for group in G if course in G[group].courses
                    ]
                )


    # ===============================
    # SHIFT CONSTRINTS
    # ===============================
    if params.is_considering_shift:
        hour_session = {}

        for course in C:
            for session in S[course]:
                hour_session[course, session] = model.NewIntVar(0, HOURS_PER_DAY - 1, f"hour_{course}_{session}")

                model.Add(
                    hour_session[course, session] ==
                    start_session[course, session] - (day_session[course, session] * HOURS_PER_DAY)
                )

        for group in G:
            shift_start = G[group].shift_start_hour
            shift_end = G[group].shift_end_hour

            for course in G[group].courses:
                dur = C[course].duration

                for session in S[course]:
                    model.Add(
                        hour_session[course, session] >= shift_start
                    ).OnlyEnforceIf(assign[group, course, session])

                    model.Add(
                        hour_session[course, session] + dur <= shift_end
                    ).OnlyEnforceIf(assign[group, course, session])


    # ===============================
    # WEEKEND CONSTRINTS
    # ===============================
    if CALENDAR.weekend_index:
        for group in G:
            if G[group].cycle == "WDays":
                for course in G[group].courses:
                    for session in S[course]:

                        for wd in CALENDAR.weekend_index:
                            model.Add(
                                day_session[course, session] != wd
                            ).OnlyEnforceIf(assign[group, course, session])


    # ===============================
    # VALID PERIOD CONSTRAINTS FOR COURSES
    # ===============================
    for course in C:
        if course in S:
            valid_start = C[course].valid_start_date
            valid_end = C[course].valid_end_date

            if valid_start:
                valid_start_day = CALENDAR.index[valid_start]

                for session in S[course]:
                    model.Add(
                        day_session[course, session] >= valid_start_day
                    ).OnlyEnforceIf(active_session[course, session])

            if valid_end:
                valid_end_day = CALENDAR.index[valid_end]

                for session in S[course]:
                    model.Add(
                        day_session[course, session] <= valid_end_day
                    ).OnlyEnforceIf(active_session[course, session])


    # ===============================
    # DAILY TRAINEE LIMIT (≤ MAX_SESSION_LENGTH)
    # ===============================
    for group in G:
        for day in range(DAYS):
            terms = []

            for course in G[group].courses:
                dur = min(C[course].duration, MAX_SESSION_LENGTH)

                for session in S[course]:

                    is_day = model.NewBoolVar(
                        f"isday_{group}_{course}_{session}_{day}"
                    )

                    model.Add(
                        day_session[course, session] == day
                    ).OnlyEnforceIf(is_day)

                    model.Add(
                        day_session[course, session] != day
                    ).OnlyEnforceIf(is_day.Not())

                    attend_today = model.NewBoolVar(
                        f"attend_{group}_{course}_{session}_{day}"
                    )

                    # attend_today = assign AND is_day
                    model.AddBoolAnd([
                        assign[group, course, session],
                        is_day
                    ]).OnlyEnforceIf(attend_today)

                    model.AddBoolOr([
                        assign[group, course, session].Not(),
                        is_day.Not()
                    ]).OnlyEnforceIf(attend_today.Not())

                    terms.append(dur * attend_today)

            model.Add(sum(terms) <= MAX_SESSION_LENGTH)


    # ===============================
    # GROUP NO-OVERLAP
    # ===============================
    for group in G:
        interval_session = []

        for course in G[group].courses:
            dur = C[course].duration

            for session in S[course]:
                interval = model.NewOptionalIntervalVar(
                    start_session[course, session],
                    dur,
                    end_session[course, session],
                    assign[group, course, session],
                    f"interval_group_{group}_{course}_{session}"
                )

                interval_session.append(interval)

        model.AddNoOverlap(interval_session)


    # ===============================
    # TRAINER NO-OVERLAP
    # ===============================
    for trainer in T:
        interval_session = []

        for course in C:
            if course in S:
                dur = C[course].duration

                for session in S[course]:
                    if (course, session, trainer) in trainer_session:
                        interval = model.NewOptionalIntervalVar(
                            start_session[course, session],
                            dur,
                            end_session[course, session],
                            trainer_session[course, session, trainer],
                            f"interval_trainer_{course}_{session}_{trainer}"
                        )

                        interval_session.append(interval)

        if interval_session:
            model.AddNoOverlap(interval_session)


    # ===============================
    # VENUE NO-OVERLAP
    # ===============================
    for venue in V:
        interval_session = []

        for course in C:
            if course in S:
                dur = C[course].duration

                for session in S[course]:
                    interval = model.NewOptionalIntervalVar(
                        start_session[course, session],
                        dur,
                        end_session[course, session],
                        venue_session[course, session, venue],
                        f"interval_venue_{course}_{session}_{venue}"
                    )

                    interval_session.append(interval)

        if interval_session:
            model.AddNoOverlap(interval_session)


    # ===============================
    # VENUE CAPACITY (CUMULATIVE)
    # ===============================
    for course in C:
        if course in S:
            for session in S[course]:
                occupancy = sum(
                    len(G[group].trainees) * assign[group, course, session]
                        for group in G if course in G[group].courses
                )

                for venue in V.values():
                    model.Add(
                        occupancy <= venue.capacity).OnlyEnforceIf(venue_session[course, session, venue.name]
                    )


    # ===============================
    # PREREQUISITES (SUBGROUP LEVEL)
    # ===============================
    for group in G:
        for course in G[group].courses:
            for prereq in C[course].prerequisites:
                
                if course in S and prereq in S:
                    for s1 in S[prereq]:
                        if (group, prereq, s1) in assign:

                            for s2 in S[course]:
                                if (group, course, s2) in assign:

                                    model.Add(
                                        start_session[prereq, s1] < start_session[course, s2]
                                    ).OnlyEnforceIf(
                                        [
                                            assign[group, prereq, s1],
                                            assign[group, course, s2]
                                        ]
                                    )


    # ===============================
    # PREREQUISITES (GLOBAL LEVEL)
    # ===============================
    if params.is_using_global_sequence:
        for course in C:
            for prereq in C[course].global_sequence:
                if prereq not in S or course not in S:
                    continue

                for s_course in S[course]:
                    for s_pre in S[prereq]:

                        model.Add(
                            end_session[prereq, s_pre] <= start_session[course, s_course]
                        ).OnlyEnforceIf(
                            [
                                active_session[prereq, s_pre],
                                active_session[course, s_course]
                            ]
                        )


    # ===============================
    # OBJECTIVES: MAXIMIZE SHARED SESSIONS + EVEN DAILY DISTRIBUTION
    # ===============================

    # # --- Minimize Open Sessions ---
    # total_open_sessions = model.NewIntVar(0, 100000, "total_open_sessions")

    # model.Add(
    #     total_open_sessions ==
    #     sum(
    #         active_session[course, session]
    #             for course in C for session in S.get(course, [])
    #     )
    # )


    # --- Minimize Daily Session Imbalance
    daily_duration = {}

    for day in range(DAYS):
        daily_duration[day] = model.NewIntVar(0, HORIZON * len(C), f"daily_dur_{day}")

        terms = []
        for course in C:
            if course in S:
                dur = C[course].duration

                for session in S[course]:
                    b = model.NewBoolVar(f"is_{course}_{session}_day_{day}")

                    model.Add(day_session[course, session] == day).OnlyEnforceIf(b)
                    model.Add(day_session[course, session] != day).OnlyEnforceIf(b.Not())

                    terms.append(dur * b)
        
        if terms:
            model.Add(
                daily_duration[day] == sum(terms)
            )

    max_daily = model.NewIntVar(0, HORIZON * len(C), "max_daily")
    min_daily = model.NewIntVar(0, HORIZON * len(C), "min_daily")

    for day in range(DAYS):
        model.Add(daily_duration[day] <= max_daily)
        model.Add(daily_duration[day] >= min_daily)

    daily_imbalance = model.NewIntVar(0, HORIZON * len(C), "daily_imbalance")
    model.Add(
        daily_imbalance == max_daily - min_daily
    )


    # --- Minimize Trainer Workload Imbalance ---
    trainer_load = {}
    for trainer in T:
        trainer_load[trainer] = model.NewIntVar(0, HORIZON, f"trainer_load_{trainer}")

        model.Add(
            trainer_load[trainer] ==
            sum(
                C[course].duration * trainer_session[course, session, trainer]
                    for course in C
                        for session in S.get(course, [])
                    if (course, session, trainer) in trainer_session
            )
        )

    max_load = model.NewIntVar(0, HORIZON, "max_trainer_load")
    min_load = model.NewIntVar(0, HORIZON, "min_trainer_load")

    for trainer in T:
        model.Add(trainer_load[trainer] <= max_load)
        model.Add(trainer_load[trainer] >= min_load)

    trainer_imbalance = model.NewIntVar(0, HORIZON, "trainer_imbalance")
    model.Add(trainer_imbalance == max_load - min_load)


    # --- Minimize Sessions on Virtual Rooms ---
    virtual_venue_list = [venue.name for venue in V.values() if venue.is_virtual]
    virtual_venue_sessions = []
    for course in C:
        if course in S:
            for session in S[course]:
                for venue in virtual_venue_list:
                    virtual_venue_sessions.append(
                        venue_session[course, session, venue]
                    )

    virtual_sessions = model.NewIntVar(0, 100000, "virtual_sessions")
    model.Add(virtual_sessions == sum(virtual_venue_sessions))


    # # --- Minimize Sessions on Weekend ---
    # if weekend_list:
    #     weekend_flags = []
    #     for course in C:
    #         for session in S[course]:
    #             for wd in weekend_list:

    #                 b = model.NewBoolVar(f"weekend_{course}_{session}_{wd}")

    #                 model.Add(day_session[course, session] == wd).OnlyEnforceIf(b)
    #                 model.Add(day_session[course, session] != wd).OnlyEnforceIf(b.Not())

    #                 # count only if session is active
    #                 model.AddImplication(b, active_session[course, session])

    #                 weekend_flags.append(b)

    #     weekend_sessions = model.NewIntVar(0, 100000, "weekend_sessions")
    #     model.Add(weekend_sessions == sum(weekend_flags))


    model.Minimize(
        # total_open_sessions * 100000 +
        daily_imbalance * 1000 +
        virtual_sessions * 100 +
        # weekend_sessions * 100 +
        trainer_imbalance
    )

    # model.maximize(
    #     total_open_sessions * 100000 -
    #     daily_imbalance * 1000 -
    #     virtual_sessions * 100 -
    #     # weekend_sessions * 100 +
    #     trainer_imbalance
    # )


    # ===============================
    # SOLVE
    # ===============================

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.max_time_in_seconds
    solver.parameters.num_search_workers = params.num_search_workers

    print("Solving starts at:", pd.Timestamp.now())

    status = solver.Solve(model)

    print("Solving ends at:", pd.Timestamp.now())

    print("Status:", solver.StatusName(status))
    print(f"Objective value: {solver.ObjectiveValue()}")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        rows = []
        for group in G:
            # for subgroup, trainees in G[group]["subgroups"].items():
            trainees = len(G[group].trainees)
            for course in G[group].courses:
                course_company = C[course].company
                course_stream = C[course].stream

                # find chosen session
                chosen_session = None
                for session in S[course]:
                    if solver.Value(assign[group, course, session]):
                        chosen_session = session
                        break

                if chosen_session is None:
                    continue

                # start/end
                start = solver.Value(start_session[course, chosen_session])
                end = solver.Value(end_session[course, chosen_session])

                start_day = start // HOURS_PER_DAY
                start_hour = start % HOURS_PER_DAY

                end_day = (end - 1) // HOURS_PER_DAY
                end_hour = (end - 1) % HOURS_PER_DAY + 1

                # calendar mapping
                date_str = CALENDAR.dates[start_day].date
                day_name = datetime.datetime.strptime(str(date_str), "%Y-%m-%d").strftime("%A")

                start_time = hour_index_to_time(start_hour, is_start=True)
                end_time = hour_index_to_time(end_hour, is_start=False)

                # venue
                venue_used = None
                for venue in V:
                    if solver.Value(venue_session[course, chosen_session, venue]):
                        venue_used = venue
                        break

                # trainer
                trainer_used = None
                for trainer in T:
                    key = (course, chosen_session, trainer)
                    if key in trainer_session and solver.Value(trainer_session[key]):
                        trainer_used = trainer
                        break

                # occupancy of session
                occupancy = sum(
                    trainees for g in G
                        if course in G[g].courses and solver.Value(assign[g, course, chosen_session])
                )

                rows.append([
                    group,
                    trainees,
                    course,
                    course_company,
                    course_stream,
                    start_day,
                    start_hour,
                    end_day,
                    end_hour,
                    date_str,
                    day_name,
                    start_time,
                    end_time,
                    venue_used,
                    V[venue_used].capacity,
                    occupancy,
                    trainer_used,
                    '-'
                ])

        df = pd.DataFrame(rows, columns=[
            "Group",
            "Trainees",
            "Course",
            "Company",
            "Stream",
            "Start Day",
            "Start Hour",
            "End Day",
            "End Hour",
            "Date",
            "Day",
            "Start Time",
            "End Time",
            "Venue",
            "Venue Max Capacity",
            "Venue Occupancy",
            "Trainer",
            "Session"
        ])

        print("\nSCHEDULE:")
        print(df)
        df.to_csv(f"export/{params.report_name}_schedule.csv", index=False)

        # # Merge only the 'trainee' column from groups_trainee_df into df
        # df = df.merge(
        #     groups_trainee[['group_name', 'subgroup_name', 'trainee']],
        #     left_on=["Group", "Subgroup"],
        #     right_on=["group_name", "subgroup_name"],
        #     how='left'
        # )
        
        # # Remove specified columns
        # columns_to_drop = ["Trainees", "Venue Occupancy", "group_name", "subgroup_name"]
        # df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # # Rename 'trainee' column to 'Trainee'
        # if 'trainee' in df.columns:
        #     df = df.rename(columns={'trainee': 'Trainee'})

        # print("\nSCHEDULE DETAIL:")
        # print(df)

        # df.to_csv(f"export/{params['report_name']}_schedule_detail.csv", index=False)

        # print("\nResult has been exported.")

        # from collections import defaultdict

        # def get_interval(course, session):
        #     start = solver.Value(start_session[course, session])
        #     end = solver.Value(end_session[course, session])
        #     return start, end


        # def overlap(a_start, a_end, b_start, b_end):
        #     return not (a_end <= b_start or b_end <= a_start)


        # # ===============================
        # # TRAINER OVERLAP CHECK
        # # ===============================
        # trainer_intervals = defaultdict(list)

        # for c in C:
        #     if c in S:
        #         for s in S[c]:
        #             for t in T:
        #                 key = (c, s, t)
        #                 if key in trainer_session and solver.Value(trainer_session[key]):
        #                     trainer_intervals[t].append((c, s, *get_interval(c, s)))

        # for t, sessions in trainer_intervals.items():
        #     for i in range(len(sessions)):
        #         for j in range(i + 1, len(sessions)):
        #             _, _, s1, e1 = sessions[i]
        #             _, _, s2, e2 = sessions[j]
        #             if overlap(s1, e1, s2, e2):
        #                 print("TRAINER OVERLAP:", t, sessions[i], sessions[j])


        # # ===============================
        # # VENUE OVERLAP CHECK
        # # ===============================
        # venue_intervals = defaultdict(list)

        # for c in C:
        #     if c in S:
        #         for s in S[c]:
        #             for v in V:
        #                 if solver.Value(venue_session[c, s, v]):
        #                     venue_intervals[v].append((c, s, *get_interval(c, s)))

        # for v, sessions in venue_intervals.items():
        #     for i in range(len(sessions)):
        #         for j in range(i + 1, len(sessions)):
        #             _, _, s1, e1 = sessions[i]
        #             _, _, s2, e2 = sessions[j]
        #             if overlap(s1, e1, s2, e2):
        #                 print("VENUE OVERLAP:", v, sessions[i], sessions[j])


        # # ===============================
        # # SUBGROUP OVERLAP CHECK
        # # ===============================
        # subgroup_intervals = defaultdict(list)

        # for g in G:
        #     for u in G[g]["subgroups"]:
        #         for c in G[g]["courses"]:
        #             for s in S[c]:
        #                 if solver.Value(assign[g, u, c, s]):
        #                     subgroup_intervals[(g, u)].append(
        #                         (c, s, *get_interval(c, s))
        #                     )

        # for gu, sessions in subgroup_intervals.items():
        #     for i in range(len(sessions)):
        #         for j in range(i + 1, len(sessions)):
        #             _, _, s1, e1 = sessions[i]
        #             _, _, s2, e2 = sessions[j]
        #             if overlap(s1, e1, s2, e2):
        #                 print("SUBGROUP OVERLAP:", gu, sessions[i], sessions[j])


        # # ===============================
        # # VENUE CAPACITY CHECK
        # # ===============================
        # for c in C:
        #     if c in S:
        #         for s in S[c]:
        #             # compute occupancy
        #             occupancy = sum(
        #                 G[g]["subgroups"][u]
        #                 for g in G
        #                 for u in G[g]["subgroups"]
        #                 if c in G[g]["courses"]
        #                 and solver.Value(assign[g, u, c, s])
        #             )

        #             for v in V:
        #                 if solver.Value(venue_session[c, s, v]):
        #                     if occupancy > venues[v]:
        #                         print(
        #                             "CAPACITY BREACH:",
        #                             c, s,
        #                             "venue", v,
        #                             "occupancy", occupancy,
        #                             "capacity", venues[v]
        #                         )


    
