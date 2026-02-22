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
    groups = read_trainees(params)

    print_group = {group.name: group.model_dump() for group in groups.values()}
    print("\n", highlight(json.dumps(print_group, indent=4), lexers.JsonLexer(), formatters.TerminalFormatter()), "\n")

    return ModelInput(
        calendar=read_calendar(params),
        venues=read_venue(params),
        trainers=read_trainers(params),
        courses=read_courses(params),
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


def read_trainers(params: ModelParams):
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

            if eligible_courses:
                trainers[trainer_id] = Trainer(
                    name=trainer_id,
                    eligible=eligible_courses
                )

        except Exception as e:
            # print(f"Error processing trainer row: {trainer_row}, error: {e}")
            continue

    print("Len Trainers:", len(trainers))

    return trainers


def read_courses(params: ModelParams):
    _df_course = pd.read_csv(params.file_master_course)
    _df_course['course_name'] = _df_course['course_name'].str.strip()
    _df_course = _df_course[_df_course['course_name']!= '']

    _df_course['stream'] = _df_course['stream'].str.strip()
    if 'duration_minutes' not in _df_course.columns:
        _df_course['duration_minutes'] = _df_course['duration']

    _df_course['duration_minutes'] = pd.to_numeric(
        _df_course['duration_minutes'], errors='coerce').fillna(params.default_course_duration*60)

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

            courses[course_name] = Course(
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

    return courses


def read_trainees(params: ModelParams):
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
                _trainees.append(
                    Trainee(
                        company=trainee_company,
                        name=trainee_name,
                        shift=trainee_shift,
                        courses=enrolled_courses,
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

    # _groups = [Group(**group) for group in _groups.values()]

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