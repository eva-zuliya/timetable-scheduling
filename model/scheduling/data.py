import pandas as pd
import math
import json
from pygments import highlight, lexers, formatters
from pydantic import BaseModel
from typing import Optional

from .schema import *
from schema import ModelParams
from .utils import *


def read_data(params: ModelParams) -> ModelInput:
    calendar = read_calendar(params)

    course_batches, course_batches_mapping = read_courses(params, calendar)

    trainers = read_trainers(params, course_batches_mapping)
    unique_trained_courses_list = list(
        dict.fromkeys(
            course
            for trainer in trainers.values()
            for course in trainer.eligible
        )
    )

    # print_trainers = {trainer.name: trainer.model_dump() for trainer in trainers.values()}
    # print("\n", highlight(json.dumps(print_trainers, indent=4), lexers.JsonLexer(), formatters.TerminalFormatter()), "\n")

    groups = read_trainees(params, course_batches_mapping)
    course_list = set(course_batches.keys())
    for key, group in groups.items():
        groups[key].courses = [x for x in group.courses if x in course_list and x in unique_trained_courses_list]

    print_group = {group.name: group.model_dump() for group in groups.values()}
    for key in list(print_group.keys()):
        print_group[key]["trainees"] = len(print_group[key]["trainees"])

    print("\n", highlight(json.dumps(print_group, indent=4), lexers.JsonLexer(), formatters.TerminalFormatter()), "\n")

    venue = read_venue(params)

    return ModelInput(
        calendar=calendar,
        venues=venue,
        trainers=trainers,
        courses=course_batches,
        groups=groups
    )


def read_venue(params: ModelParams):
    _df_venue = pd.read_csv(params.file_master_venue)

    if params.companies:
        _df_venue = _df_venue[_df_venue['company'].isin(params.companies)]

    venues = {}
    for _, row in _df_venue.iterrows():
        name, capacity, company = row['venue_name'], row['capacity'], row['company']
        is_virtual = row['is_virtual'] if 'is_virtual' in row else False

        venues[name] = Venue(
            company=company,
            name=name,
            capacity=capacity+params.buffer_capacity,
            is_virtual=is_virtual
        )

    print("Len Venues:", len(venues))
    
    return venues


def read_courses(params: ModelParams, calendar: Calendar):
    _df_course = pd.read_csv(params.file_master_course)
    _df_course['course_name'] = _df_course['course_name'].str.strip()
    _df_course = _df_course[_df_course['course_name']!= '']

    _df_course['stream'] = _df_course['stream'].str.strip()
    if 'duration_minutes' not in _df_course.columns:
        _df_course['duration_minutes'] = _df_course['duration']

    _df_course['duration_minutes'] = pd.to_numeric(
        _df_course['duration_minutes'], errors='coerce').fillna(params.default_course_duration*60)

    # If duration_minutes < 0, replace with default_course_duration*60 as well
    _df_course.loc[_df_course['duration_minutes'] < 0, 'duration_minutes'] = params.default_course_duration * 60

    _df_prereq = pd.read_csv(params.file_master_course_sequence)
    _df_prereq = _df_prereq[
        _df_prereq['prerequisite_course_name'].notna() &
        (_df_prereq['prerequisite_course_name'].str.strip() != "")
    ]

    _df_prereq['course_name'] = _df_prereq['course_name'].str.strip()
    _df_prereq['prerequisite_course_name'] = _df_prereq['prerequisite_course_name'].str.strip()
    
    if 'is_global_sequence' not in _df_prereq.columns:
        _df_prereq['is_global_sequence'] = False

    if params.course_stream is not None:
        _df_course = _df_course[_df_course["stream"].isin(params.course_stream)]
    

    list_course_name = _df_course['course_name'].unique().tolist()
    _df_prereq = _df_prereq[
        (_df_prereq['course_name'].isin(list_course_name)) &
        (_df_prereq['prerequisite_course_name'].isin(list_course_name))
    ]

    courses = {}
    for _, course_row in _df_course.iterrows():
        try:
            course_name = course_row['course_name']
            course_stream = course_row['stream']
            course_company = course_row['company']
            duration = min(math.ceil(course_row['duration_minutes'] / 60), params.hours_per_day)

            prereqs = _df_prereq[_df_prereq['course_name'] == course_name]
            prerequisites = [] if prereqs.empty else prereqs['prerequisite_course_name'].drop_duplicates().tolist()

            seq = _df_prereq[(_df_prereq['course_name'] == course_name) & (_df_prereq['is_global_sequence'])]
            sequence = [] if seq.empty else seq['prerequisite_course_name'].drop_duplicates().tolist()

            courses[course_company, course_name] = Course(
                company=course_company,
                name=course_name,
                stream=course_stream,
                duration=duration,
                prerequisites=prerequisites,
                global_sequence=sequence
            )

        except Exception as e:
            # print(f"Error processing course row: {course_row}, error: {e}")
            continue
    
    batches_mapping = {}
    default_batch = {"batch_no": 1, "week1": 0, "week2": 0, "week3": 0, "week4": 0}

    if params.file_master_course_batch is None:
        for course in courses.values():
            batches_mapping[course.company, course.name] = [default_batch]
    
    else:
        dfs = []
        for file in params.file_master_course_batch:
            dfs.append(pd.read_csv(file))

        _df_batch = pd.concat(dfs, ignore_index=True)
        _df_batch['course_name'] = _df_batch['course_name'].str.strip()

        for (company, course, batch), group in _df_batch.groupby(
            ["company", "course_name", "batch_no"]
        ):
            row = group.iloc[0]
            key = (company, course)
            batch_dict = {
                "batch_no": batch,
                "week1": row["week1"],
                "week2": row["week2"],
                "week3": row["week3"],
                "week4": row["week4"],
            }

            batches_mapping.setdefault(key, []).append(batch_dict)


    week_groups = Calendar.week_groups

    course_batches: dict[str, CourseBatch] = {}
    course_batches_mapping: dict[str, list[str]] = {}
    for course in courses.values():
        course_batches_mapping[course.name] = []
        batches = batches_mapping.get((course.company, course.name), [default_batch])

        for batch_info in batches:
            w1, w2, w3, w4 = batch_info["week1"], batch_info["week2"], batch_info["week3"], batch_info["week4"]

            valid_start_domain = week_to_horizon_slots(
                week_groups,
                {
                    0:w1,
                    1:w2,
                    2:w3,
                    3:w4
                },
                params.hours_per_day
            )

            batch = CourseBatch(
                **course.model_dump(),
                batch_number=batch_info["batch_no"],
                valid_start_domain=valid_start_domain
            )

            course_batches[batch.id] = batch
            course_batches_mapping[course.name].append(batch.id)

    # Updating prerequsites and global sequence with batch id
    for batch_id, batch in course_batches.items():
        prerequisites = batch.prerequisites
        if prerequisites is not None:
            batch_prerequisites = [item for k in prerequisites for item in course_batches_mapping[k]]
            course_batches[batch_id].prerequisites = batch_prerequisites

        sequence = batch.global_sequence
        if sequence is not None:
            batch_sequence = [item for k in sequence for item in course_batches_mapping[k]]
            course_batches[batch_id].global_sequence = batch_sequence

    print("Len Course Batches:", len(course_batches))

    return course_batches, course_batches_mapping


def read_trainers(params: ModelParams, course_batches_mapping: dict[str, list[str]]):
    _df_trainer = pd.read_csv(params.file_master_trainer)
    _df_trainer = _df_trainer.drop_duplicates(subset=["trainer_id"])
    _df_trainer['trainer_id'] = _df_trainer['trainer_id'].astype(str)
    _df_trainer = _df_trainer[_df_trainer['trainer_id'] != '']
    
    _df_eligible = pd.read_csv(params.file_master_course_trainer)
    _df_eligible['trainer_id'] = _df_eligible['trainer_id'].astype(str)

    trainers = {}
    for _, trainer_row in _df_trainer.iterrows():
        try:
            trainer_id = trainer_row['trainer_id']
            eligible_courses = _df_eligible[_df_eligible['trainer_id'] == trainer_id]['course_name'].drop_duplicates().tolist()
            eligible_course_batches = [item for k in eligible_courses for item in course_batches_mapping[k]]

            if eligible_courses:
                trainers[trainer_id] = Trainer(
                    name=trainer_id,
                    eligible=eligible_course_batches
                )

        except Exception as e:
            # print(f"Error processing trainer row: {trainer_row}, error: {e}")
            continue

    print("Len Trainers:", len(trainers))

    return trainers


def read_trainees(params: ModelParams, course_batches_mapping: dict[str, list[str]]):
    _df_trainee = pd.read_csv(params.file_master_trainee)
    _df_trainee['employee_id'] = _df_trainee['employee_id'].astype(str)
    _df_trainee = _df_trainee.drop_duplicates(subset=["employee_id"])
    _df_trainee = _df_trainee[_df_trainee['employee_id'].astype(str).str.strip() != '']

    if params.companies is not None:
        _df_trainee = _df_trainee[_df_trainee['company'].isin(params.companies)]


    if 'is_available_saturday' not in _df_trainee.columns:
        _df_trainee['is_available_saturday'] = False
        
    _df_trainee['cycle'] = _df_trainee['is_available_saturday'].apply(lambda x: "WEnd" if x else "WDays")
    _df_trainee['cycle'] = 'WEnd'

    _df_enrollment = pd.read_csv(params.file_master_course_trainee)
    _df_enrollment = _df_enrollment[
        _df_enrollment["course_name"].isin(
            _df_enrollment.groupby("course_name")["employee_id"].nunique().loc[lambda s: s >= params.minimum_course_participant].index
        )
    ]

    _df_enrollment['employee_id'] = _df_enrollment['employee_id'].astype(str)
    _df_enrollment['course_name'] = _df_enrollment['course_name'].str.strip()
    _df_enrollment = _df_enrollment[_df_enrollment['course_exist'] == 'TRUE']

    if params.course_stream is not None:
        _df_course = pd.read_csv(params.file_master_course)
        _df_course = _df_course[_df_course["stream"].isin(params.course_stream)]

        _course_list = _df_course['course_name'].drop_duplicates().tolist()
        _df_enrollment = _df_enrollment[_df_enrollment["course_name"].isin(_course_list)]

    
    if params.file_master_course_batch is not None:
        dfs = []
        for file in params.file_master_course_batch:
            dfs.append(pd.read_csv(file))

        _df_batch = pd.concat(dfs, ignore_index=True)
        _df_batch['course_name'] = _df_batch['course_name'].astype(str).str.strip()
        _df_batch['trainee_id'] = _df_batch['trainee_id'].astype(str).str.strip()
        
        batch_lookup = _df_batch.set_index(["company", "course_name", "trainee_id"])["batch_no"]


    _trainees = []
    for _, trainee_row in _df_trainee.iterrows():
        try:
            trainee_name = trainee_row['employee_id']
            trainee_cycle = trainee_row['cycle']
            trainee_company = trainee_row['company']

            if params.is_considering_shift:
                trainee_shift = trainee_row['shift']
                if pd.isna(trainee_shift) or str(trainee_shift).strip() == "":
                    trainee_shift = "Non Shift"
            else:
                trainee_shift = "NS"

            # Get courses for this trainee from enrollment dataframe
            enrolled_courses = _df_enrollment[_df_enrollment['employee_id'] == trainee_name]['course_name'].drop_duplicates().tolist()

            if enrolled_courses:  # Only include trainees with at least one course
                enrolled_courses_batches = []
                for course in enrolled_courses:
                    if params.file_master_course_batch is None:
                        batch_no = 1
                    
                    else:
                        batch_no = batch_lookup.get((trainee_company, course, trainee_name), 1)
                    
                    enrolled_courses_batches.append(
                        f"[{trainee_company}]-[{course}]-[{batch_no}]"
                    )

                _trainees.append(
                    Trainee(
                        company=trainee_company,
                        name=trainee_name,
                        shift=trainee_shift,
                        courses=enrolled_courses_batches,
                        cycle=trainee_cycle
                    )
                )
                # print("Added trainee:", trainee_name, "Shift:", trainee_shift, "Cycle:", trainee_cycle, "Courses:", enrolled_courses)

        except Exception as e:
            # print(f"Error processing trainee row: {trainee_row}, error: {e}")
            continue


    _groups = {}
    for trainee in _trainees:
        course_key = tuple(sorted(trainee.courses))
        # group_key = tuple(list(course_key) + [trainee.shift, trainee.cycle])
        group_key = tuple(list(course_key))
        
        if group_key not in _groups:
            _groups[group_key] = {
                "name": f"G{len(_groups) + 1}",
                "courses": list(course_key),
                "trainees": [],
                "shift": trainee.shift,
                "shift_start_hour": trainee.shift_start_hour,
                "shift_end_hour": trainee.shift_end_hour,
                "cycle": trainee.cycle
            }

        _groups[group_key]["trainees"].append(trainee.name)

    groups = {}
    for value in _groups.values():
        group = Group(**value)
        groups[group.name] = group

    print("Len Trainees:", len(_trainees), "\nLen Groups:", len(groups))

    return groups


def read_calendar(params: ModelParams):
    return Calendar(
        start_date=params.start_date,
        days=params.days
    )