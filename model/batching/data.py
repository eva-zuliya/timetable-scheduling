from .schema import *
from schema import ModelParams
import pandas as pd
from pygments import highlight, lexers, formatters
import json


def read_data(params: ModelParams, company: str) -> ModelInput:

    courses = read_courses(params, company)
    shifts = read_trainees(params, company)

    print_group = {group.name: group.model_dump() for group in courses.values()}
    for key in list(print_group.keys()):
        print_group[key]["trainees"] = len(print_group[key]["trainees"])
        print_group[key]["max_batches"] = courses[key].max_batches

    print("\n", highlight(json.dumps(print_group, indent=4), lexers.JsonLexer(), formatters.TerminalFormatter()), "\n")

    print("Len Courses: ", len(courses))

    return ModelInput(
        courses=courses,
        shifts=shifts
    )


def read_courses(params: ModelParams, company: str):
    _df_venue = pd.read_csv(params.file_master_venue)
    _df_venue = _df_venue[_df_venue['company']==company]
    max_venue_capacity_available = _df_venue['capacity'].max()

    _df_trainee = pd.read_csv(params.file_master_trainee)
    _df_trainee['employee_id'] = _df_trainee['employee_id'].astype(str)
    _df_trainee = _df_trainee.drop_duplicates(subset=["employee_id"])
    _df_trainee = _df_trainee[_df_trainee['employee_id'].astype(str).str.strip() != '']
    _df_trainee = _df_trainee[_df_trainee['company']==company]
    trainee_list = _df_trainee['employee_id'].unique().tolist()
    print("Len Trainee List: ", len(trainee_list))


    _df_enrollment = pd.read_csv(params.file_master_course_trainee)
    _df_enrollment['employee_id'] = _df_enrollment['employee_id'].astype(str)
    _df_enrollment['course_name'] = _df_enrollment['course_name'].str.strip()
    _df_enrollment = _df_enrollment[_df_enrollment['course_exist'] == True]

    if params.course_stream is not None:
        _df_course = pd.read_csv(params.file_master_course)
        _df_course = _df_course[_df_course["stream"].isin(params.course_stream)]

        _course_list = _df_course['course_name'].drop_duplicates().tolist()
        _df_enrollment = _df_enrollment[_df_enrollment["course_name"].isin(_course_list)]

    _df_enrollment = _df_enrollment[_df_enrollment["employee_id"].isin(trainee_list)]


    _df_trainer = pd.read_csv(params.file_master_trainer)
    _df_trainer = _df_trainer.drop_duplicates(subset=["trainer_id"])
    _df_trainer['trainer_id'] = _df_trainer['trainer_id'].astype(str)
    _df_trainer = _df_trainer[_df_trainer['trainer_id'] != '']
    trainer_list = _df_trainer['trainer_id'].unique().tolist()
    
    _df_eligible = pd.read_csv(params.file_master_course_trainer)
    _df_eligible['trainer_id'] = _df_eligible['trainer_id'].astype(str)
    _df_eligible = _df_eligible[_df_eligible["trainer_id"].isin(trainer_list)]


    course_list = _df_enrollment['course_name'].unique().tolist()
    print("Len Course List: ", len(course_list))
    
    courses = {}
    for course in course_list:
        trainees = _df_enrollment[_df_enrollment['course_name'] == course]['employee_id'].unique().tolist()
        trainer = _df_eligible[_df_eligible['course_name'] == course]['trainer_id'].unique().tolist()

        if trainees and trainer:
            courses[course] = CourseStats(
                company=company,
                name=course,
                trainees=trainees,
                count_trainee=len(trainees),
                count_trainers=len(trainer),
                max_venue_capacity_available=max_venue_capacity_available
            )

    return courses


def read_trainees(params: ModelParams, company: str):
    _df_trainee = pd.read_csv(params.file_master_trainee)
    _df_trainee['employee_id'] = _df_trainee['employee_id'].astype(str)
    _df_trainee = _df_trainee.drop_duplicates(subset=["employee_id"])
    _df_trainee = _df_trainee[_df_trainee['employee_id'].astype(str).str.strip() != '']
    _df_trainee = _df_trainee[_df_trainee['company']==company]

    _df_enrollment = pd.read_csv(params.file_master_course_trainee)
    _df_enrollment['employee_id'] = _df_enrollment['employee_id'].astype(str)
    _df_enrollment['course_name'] = _df_enrollment['course_name'].str.strip()
    _df_enrollment = _df_enrollment[_df_enrollment['course_exist'] == 'TRUE']

    if params.course_stream is not None:
        _df_course = pd.read_csv(params.file_master_course)
        _df_course = _df_course[_df_course["stream"].isin(params.course_stream)]

        _course_list = _df_course['course_name'].drop_duplicates().tolist()
        _df_enrollment = _df_enrollment[_df_enrollment["course_name"].isin(_course_list)]
    
    mapping = {
        "Shift 1": 1,
        "Shift 2": 2,
        "Shift 3": 3
    }

    cols = ["shift_w1", "shift_w2", "shift_w3", "shift_w4"]
    _df_trainee[cols] = _df_trainee[cols].apply(
        lambda s: s.astype(str).map(mapping).fillna(0).astype(int)
    )

    shifts = {}
    for _, trainee_row in _df_trainee.iterrows():
        try:
            trainee_name = trainee_row['employee_id']

            shift_w1 = trainee_row['shift_w1']
            shift_w2 = trainee_row['shift_w2']
            shift_w3 = trainee_row['shift_w3']
            shift_w4 = trainee_row['shift_w4']
            
            # Get courses for this trainee from enrollment dataframe
            enrolled_courses = _df_enrollment[_df_enrollment['employee_id'] == trainee_name]['course_name'].drop_duplicates().tolist()

            if enrolled_courses:  # Only include trainees with at least one course
                shifts[trainee_name] = TraineeShift(
                    name=trainee_name,
                    week1=shift_w1,
                    week2=shift_w2,
                    week3=shift_w3,
                    week4=shift_w4
                )
  
        except Exception as e:
            # print(f"Error processing trainee row: {trainee_row}, error: {e}")
            continue

    return shifts