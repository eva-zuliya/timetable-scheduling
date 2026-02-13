import calendar
import pandas as pd
import math
import json
from pygments import highlight, lexers, formatters
from schema import Venue, Trainer, Course, Trainee, Group, Calendar
from utils import export_groups_courses_to_df, export_groups_trainee_to_df


def read_data(params: dict):
    venues, venue_list = read_venue(
        file_master_venue=params['file_master_venue']
    )

    trainers, eligible = read_trainers(
        file_master_trainer=params['file_master_trainer'],
        file_master_course_trainer=params['file_master_course_trainer']
    )

    courses = read_courses(
        file_master_course=params['file_master_course'],
        file_master_course_sequence=params['file_master_course_sequence']
    )

    groups = read_trainees(
        file_master_trainee=params['file_master_trainee'],
        file_master_course_trainee=params['file_master_course_trainee'],
        report_name=params['report_name'],
        minimum_course_participant=params['minimum_course_participant'],
        maximum_group_size=params['maximum_group_size']
    )

    weekend_list = read_calendar(
        start_date=params['start_date'],
        days=params['days']
    )

    print("\n", highlight(json.dumps(groups, indent=4), lexers.JsonLexer(), formatters.TerminalFormatter()), "\n")

    return {
        'days': params['days'],
        'hours_per_day': params['hours_per_day'],
        'horizon': params['days'] * params['hours_per_day'],
        'max_session_length': params['maximum_session_length'],
        'venues': venues,
        'venue_list': venue_list,
        'trainers': trainers,
        'eligible': eligible,
        'courses': courses,
        'groups': groups,
        'is_considering_shift': params['is_considering_shift'],
        'weekend_list': weekend_list
    }


def read_venue(
    file_master_venue: str
):
    _df_venue = pd.read_csv(file_master_venue)
    _venues = [
        Venue(name=row['venue_name'], capacity=row['capacity']+5)
        for _, row in _df_venue.iterrows()
    ]

    venues = {venue.name: venue.capacity for venue in _venues}
    venue_list = list(venues.keys())

    print("Len Venues:", len(venues), "Len Venue List:", len(venue_list))

    return venues, venue_list


def read_trainers(
    file_master_trainer: str,
    file_master_course_trainer: str
):
    _df_trainer = pd.read_csv(file_master_trainer)
    _df_trainer = _df_trainer.drop_duplicates(subset=["trainer_id"])

    _df_eligible = pd.read_csv(file_master_course_trainer)

    _trainers = []
    for _, trainer_row in _df_trainer.iterrows():
        trainer_id = trainer_row['trainer_id']
        trainer_name = trainer_row['trainer_name']
        
        # Get eligible courses for this trainer from eligibility dataframe
        eligible_courses = _df_eligible[_df_eligible['trainer_id'] == trainer_id]['course_name'].tolist()

        if eligible_courses:  # Only include trainers with at least one eligible course
            _trainers.append(Trainer(name=trainer_name, eligible=eligible_courses))

    eligible = {(trainer.name, course): 1 for trainer in _trainers for course in trainer.eligible}
    trainers = [trainer.name for trainer in _trainers]

    print("Len Eligible:", len(eligible), "Len Trainers:", len(trainers))

    return trainers, eligible


def read_courses(
    file_master_course: str,
    file_master_course_sequence: str
):
    _df_course = pd.read_csv(file_master_course)

    _df_prereq = pd.read_csv(file_master_course_sequence)
    # _df_prereq = _df_prereq[~(_df_prereq['is_course_valid'].str.contains('INVALID', na=False) | 
    #                         _df_prereq['is_prerequisite_valid'].str.contains('INVALID', na=False))]
    _df_prereq = _df_prereq[
        _df_prereq['prerequisite_course_name'].notna() &
        (_df_prereq['prerequisite_course_name'].str.strip() != "")
    ]

    # Build courses from CSV data
    _courses = []
    for _, course_row in _df_course.iterrows():
        course_name = course_row['course_name']
        duration = math.ceil(course_row['duration_minutes'] / 60)  

        # Get prerequisites for this course from prerequisite dataframe
        prereqs = _df_prereq[_df_prereq['course_name'] == course_name]
        prerequisites = [] if prereqs.empty else prereqs['prerequisite_course_name'].tolist()

        _courses.append(Course(name=course_name, duration=duration, prerequisites=prerequisites))

    courses = {
        course.name: {
            "dur": course.duration,
            "prereq": course.prerequisites
        } for course in _courses
    }

    print("Len Courses:", len(courses))

    return courses


def read_trainees(
    file_master_trainee: str,
    file_master_course_trainee: str,
    report_name: str = 'report',
    minimum_course_participant: int = None,
    maximum_group_size: int = 30,
    is_considering_shift: bool = False
):
    _df_trainee = pd.read_csv(file_master_trainee)
    _df_trainee = _df_trainee.drop_duplicates(subset=["employee_id"])

    _df_enrollment = pd.read_csv(file_master_course_trainee)
    _df_enrollment = _df_enrollment[
        _df_enrollment["course_name"].isin(
            _df_enrollment.groupby("course_name")["employee_id"].nunique().loc[lambda s: s >= minimum_course_participant].index
        )
    ]

    _trainees = []
    for _, trainee_row in _df_trainee.iterrows():
        trainee_name = trainee_row['employee_id']

        if is_considering_shift:
            trainee_shift = trainee_row['shift']
            if pd.isna(trainee_shift) or str(trainee_shift).strip() == "":
                trainee_shift = "Non Shift"
        else:
            trainee_shift = "NS"

        # Get courses for this trainee from enrollment dataframe
        enrolled_courses = _df_enrollment[_df_enrollment['employee_id'] == trainee_name]['course_name'].tolist()

        if enrolled_courses:  # Only include trainees with at least one course
            _trainees.append(
                Trainee(
                    name=trainee_name,
                    shift=trainee_shift,
                    courses=enrolled_courses,
                    cycle="WDays"  # Later should be based on the master employee data
                )
            )

    _groups = {}
    for trainee in _trainees:
        course_key = tuple(sorted(trainee.courses))
        group_key = tuple(list(course_key) + [trainee.shift, trainee.cycle])
        
        if group_key not in _groups:
            _groups[group_key] = {
                "name": f"G{len(_groups) + 1} - {trainee.shift} - {trainee.cycle}",
                "courses": list(course_key),
                "trainees": [],
                "shift": trainee.shift,
                "shift_start_hour": trainee.shift_start_hour,
                "shift_end_hour": trainee.shift_end_hour,
                "cycle": trainee.cycle
            }

        _groups[group_key]["trainees"].append(trainee.name)

    _groups = [Group(**group) for group in _groups.values()]

    for group in _groups:
        group.split_subgroups(maximum_group_size)

    export_groups_trainee_to_df(groups=_groups, report_name=report_name)
    export_groups_courses_to_df(groups=_groups, report_name=report_name)

    groups = {
        group.name: {
            "shift_start_hour": group.shift_start_hour,
            "shift_end_hour": group.shift_end_hour,
            "cycle": group.cycle,
            "courses": group.courses,
            "subgroups": {subgroup: len(members) for subgroup, members in (group.subgroup or {}).items()}
        } for group in _groups
    }

    print("Len Trainees:", len(_trainees), "Len Groups:", len(groups))

    return groups


def read_calendar(
    start_date: str,
    days: int
):
    calendar = Calendar(
        start_date=start_date,
        days = days
    )

    return calendar.weekend_index