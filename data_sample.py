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

    groups, group_trainee = read_trainees(
        file_master_trainee=params['file_master_trainee'],
        file_master_course_trainee=params['file_master_course_trainee'],
        report_name=params['report_name'],
        minimum_course_participant=params['minimum_course_participant'],
        maximum_group_size=params['maximum_group_size']
    )

    calendar, weekend_list = read_calendar(
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
        'groups_trainee': group_trainee,
        'is_considering_shift': params['is_considering_shift'],
        'calendar': calendar,
        'weekend_list': weekend_list
    }


def read_venue(
    file_master_venue: str
):
    _venues = [
        Venue(name="V1", capacity=5),
        Venue(name="V2", capacity=5),
        Venue(name="V3", capacity=5),
        Venue(name="V4", capacity=5),
        Venue(name="V5", capacity=5)
    ]

    venues = {venue.name: venue.capacity for venue in _venues}
    venue_list = list(venues.keys())

    print("Len Venues:", len(venues), "Len Venue List:", len(venue_list))

    return venues, venue_list


def read_trainers(
    file_master_trainer: str,
    file_master_course_trainer: str
):
    _trainers = [
        Trainer(name="T1", eligible=["C1", "C2"]),
        Trainer(name="T2", eligible=["C1", "C2", "C3"]),
        Trainer(name="T3", eligible=["C1", "C2", "C3"]),
        Trainer(name="T4", eligible=["C1", "C2", "C3"]),
    ]

    eligible = {(trainer.name, course): 1 for trainer in _trainers for course in trainer.eligible}
    trainers = [trainer.name for trainer in _trainers]

    print("Len Eligible:", len(eligible), "Len Trainers:", len(trainers))

    return trainers, eligible


def read_courses(
    file_master_course: str,
    file_master_course_sequence: str
):
    _courses = [
        Course(name="C1", duration=4, prerequisites=[]),
        Course(name="C2", duration=4, prerequisites=["C1"]),
        Course(name="C3", duration=4, prerequisites=["C1", "C2"]),
    ]

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
    maximum_group_size: int = 30
):

    _trainees = [
        Trainee(name="E1", courses=["C1", "C2"], shift="Shift 1", cycle="WDays"),
        Trainee(name="E2", courses=["C1", "C2"], shift="Shift 1", cycle="WDays"),
        Trainee(name="E3", courses=["C1", "C2"], shift="Non Shift", cycle="WDays"),
        Trainee(name="E4", courses=["C1", "C2"], shift="Non Shift", cycle="WDays"),
        Trainee(name="E5", courses=["C1", "C2"], shift="Non Shift", cycle="WDays"),
        Trainee(name="E6", courses=["C1", "C2"], shift="Non Shift", cycle="WDays"),
        Trainee(name="E7", courses=["C1", "C2"], shift="Shift 2", cycle="WEnd"),
        Trainee(name="E8", courses=["C1", "C2", "C3"], shift="Shift 2", cycle="WEnd")
    ]

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

    _df_group_trainee = export_groups_trainee_to_df(groups=_groups, report_name=report_name)
    _df_group_courses = export_groups_courses_to_df(groups=_groups, report_name=report_name)

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

    return groups, _df_group_trainee


def read_calendar(
    start_date: str,
    days: int
):
    calendar = Calendar(
        start_date=start_date,
        days = days
    )

    return calendar, calendar.weekend_index