from schema import Venue, Trainer, Course, Trainee, Group
import pandas as pd

# ===============================
# TIME AND PARAMS
# ===============================
DAYS = 10
HOURS_PER_DAY = 10
HORIZON = DAYS * HOURS_PER_DAY

MAX_SESSION_LENGTH = 8
MAX_GROUP_SIZE = 2


# ===============================
# VENUES (capacity)
# ===============================
_venues = [
    Venue(name="V1", capacity=5),
    Venue(name="V2", capacity=5),
    Venue(name="V3", capacity=5),
    Venue(name="V4", capacity=5)
]

venues = {venue.name: venue.capacity for venue in _venues}
venue_list = list(venues.keys())


# ===============================
# TRAINERS AND ELIGIBILITY
# ===============================
_trainers = [
    Trainer(name="T1", eligible=["C1", "C2"]),
    Trainer(name="T2", eligible=["C1", "C2", "C3"]),
]

eligible = {(trainer.name, course): 1 for trainer in _trainers for course in trainer.eligible}
trainers = [trainer.name for trainer in _trainers]


# ===============================
# COURSES
# ===============================
_courses = [
    Course(name="C1", duration=8, prerequisites=[]),
    Course(name="C2", duration=4, prerequisites=["C1"]),
    Course(name="C3", duration=4, prerequisites=["C1", "C2"]),
]

courses = {
    course.name: {
        "dur": course.duration,
        "prereq": course.prerequisites
    } for course in _courses
}


# ===============================
# TRAINEE AND GROUPS → SUBGROUPS
# ===============================
_trainees = [
    Trainee(name="E1", courses=["C1", "C2"]),
    Trainee(name="E2", courses=["C1", "C2"]),
    Trainee(name="E3", courses=["C1", "C2"]),
    Trainee(name="E4", courses=["C1", "C2"]),
    Trainee(name="E5", courses=["C1", "C2"]),
    Trainee(name="E6", courses=["C1", "C2"]),
    Trainee(name="E7", courses=["C1", "C2"]),
    Trainee(name="E8", courses=["C1", "C2", "C3"])
]

_groups = {}
for trainee in _trainees:
    course_key = tuple(sorted(trainee.courses))  # use tuple as key since set is unhashable
    if course_key not in _groups:
        _groups[course_key] = {
            "name": "__".join(course_key),
            "courses": list(course_key),
            "trainees": []
        }
    _groups[course_key]["trainees"].append(trainee.name)

_groups = [Group(**group) for group in _groups.values()]

for group in _groups:
    group.split_subgroups(MAX_GROUP_SIZE)

groups = {
    group.name: {
        "courses": group.courses,
        "subgroups": {subgroup: len(members) for subgroup, members in (group.subgroup or {}).items()}
    } for group in _groups
}
