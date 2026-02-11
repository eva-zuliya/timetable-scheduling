from schema import Venue, Trainer, Course, Trainee, Group, export_groups_to_df
import pandas as pd
import math

# ===============================
# TIME AND PARAMS
# ===============================
DAYS = 28
HOURS_PER_DAY = 8
HORIZON = DAYS * HOURS_PER_DAY

MAX_SESSION_LENGTH = 6
MAX_GROUP_SIZE = 30


# ===============================
# VENUES (capacity)
# ===============================
# _venues = [
#     Venue(name="V1", capacity=5),
#     Venue(name="V2", capacity=5),
#     Venue(name="V3", capacity=5),
#     Venue(name="V4", capacity=5)
# ]

_df_venue = pd.read_csv('data_source/[Data] Master Training Scheduling - PAS - Master Venue.csv')
_venues = [
    Venue(name=row['venue_name'], capacity=row['capacity']+5)
    for _, row in _df_venue.iterrows()
]

venues = {venue.name: venue.capacity for venue in _venues}
venue_list = list(venues.keys())

print("Len Venues:", len(venues), "Len Venue List:", len(venue_list))

# # ===============================
# # TRAINERS AND ELIGIBILITY
# # ===============================
# _trainers = [
#     Trainer(name="T1", eligible=["C1", "C2"]),
#     Trainer(name="T2", eligible=["C1", "C2", "C3"]),
# ]

_df_trainer = pd.read_csv('data_source/[Data] Master Training Scheduling - PAS - Master Trainer.csv')
_df_trainer = _df_trainer.drop_duplicates(subset=["trainer_id"])

_df_eligible = pd.read_csv('data_source/[Data] Master Training Scheduling - PAS - Master Course Trainer.csv')

# Build trainers from CSV data
_trainers = []
for _, trainer_row in _df_trainer.iterrows():
    trainer_id = trainer_row['trainer_id']
    trainer_name = trainer_row['trainer_name']
    
    # Get eligible courses for this trainer from eligibility dataframe
    eligible_courses = _df_eligible[_df_eligible['trainer_id'] == trainer_id]['course_name'].tolist()
    
    for i in range(5):  # Duplicate each trainer 10 times with unique names
        _trainers.append(Trainer(name=f"{trainer_name}_{i+1}", eligible=eligible_courses))

    # _trainers.append(Trainer(name=trainer_name, eligible=eligible_courses))

eligible = {(trainer.name, course): 1 for trainer in _trainers for course in trainer.eligible}
trainers = [trainer.name for trainer in _trainers]

print("Len Eligible:", len(eligible), "Len Trainers:", len(trainers))


# # ===============================
# # COURSES
# # ===============================
# _courses = [
#     Course(name="C1", duration=8, prerequisites=[]),
#     Course(name="C2", duration=4, prerequisites=["C1"]),
#     Course(name="C3", duration=4, prerequisites=["C1", "C2"]),
# ]

_df_course = pd.read_csv('data_source/[Data] Master Training Scheduling - PAS - Master Course.csv')
_df_prereq = pd.read_csv('data_source/[Data] Master Training Scheduling - PAS - Master Course Sequence.csv')
_df_prereq = _df_prereq[~(_df_prereq['is_course_valid'].str.contains('INVALID', na=False) | 
                         _df_prereq['is_prerequisite_valid'].str.contains('INVALID', na=False))]
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

# # ===============================
# # TRAINEE AND GROUPS → SUBGROUPS
# # ===============================
# _trainees = [
#     Trainee(name="E1", courses=["C1", "C2"]),
#     Trainee(name="E2", courses=["C1", "C2"]),
#     Trainee(name="E3", courses=["C1", "C2"]),
#     Trainee(name="E4", courses=["C1", "C2"]),
#     Trainee(name="E5", courses=["C1", "C2"]),
#     Trainee(name="E6", courses=["C1", "C2"]),
#     Trainee(name="E7", courses=["C1", "C2"]),
#     Trainee(name="E8", courses=["C1", "C2", "C3"])
# ]

_df_trainee = pd.read_csv('data_source/[Data] Master Training Scheduling - PAS - Master Employee.csv')
_df_trainee = _df_trainee.drop_duplicates(subset=["employee_id"])

_df_enrollment = pd.read_csv('data_source/[Data] Master Training Scheduling - PAS - Master Course Employee.csv')
_df_enrollment = _df_enrollment[
    _df_enrollment["course_name"].isin(
        _df_enrollment.groupby("course_name")["employee_id"].nunique().loc[lambda s: s >= 10].index
    )
]

_trainees = []
for _, trainee_row in _df_trainee.iterrows():
    trainee_name = trainee_row['employee_id']
    # Get courses for this trainee from enrollment dataframe
    enrolled_courses = _df_enrollment[_df_enrollment['employee_id'] == trainee_name]['course_name'].tolist()

    if enrolled_courses:  # Only include trainees with at least one course
        _trainees.append(Trainee(name=trainee_name, courses=enrolled_courses))

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

export_groups_to_df(_groups)

groups = {
    group.name: {
        "courses": group.courses,
        "subgroups": {subgroup: len(members) for subgroup, members in (group.subgroup or {}).items()}
    } for group in _groups
}

print("Len Trainees:", len(_trainees), "Len Groups:", len(groups))