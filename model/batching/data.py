from .schema import *
from schema import ModelParams

def read_data(params: ModelParams, company: str) -> ModelInput:

    courses = {
        "A": CourseStats(
            company="TEST",
            name="A",
            trainees=["E1","E2","E3","E4"],
            count_trainee=4,
            count_trainers=1,
            max_venue_capacity_available=10
        ),
        "B": CourseStats(
            company="TEST",
            name="B",
            trainees=["E5","E6","E7","E8","E9","E1"],
            count_trainee=5,
            count_trainers=3,
            max_venue_capacity_available=10
        )
    }

    shifts = {
        "E1": TraineeShift(name="E1", week1=1, week2=2, week3=1, week4=2),
        "E2": TraineeShift(name="E2", week1=1, week2=2, week3=1, week4=2),
        "E3": TraineeShift(name="E3", week1=2, week2=2, week3=1, week4=1),
        "E4": TraineeShift(name="E4", week1=3, week2=1, week3=1, week4=2),
        "E5": TraineeShift(name="E5", week1=1, week2=1, week3=2, week4=2),
        "E6": TraineeShift(name="E6", week1=1, week2=1, week3=2, week4=2),
        "E7": TraineeShift(name="E7", week1=2, week2=2, week3=2, week4=1),
        "E8": TraineeShift(name="E8", week1=2, week2=2, week3=2, week4=1),
        "E9": TraineeShift(name="E9", week1=1, week2=2, week3=1, week4=2),
    }

    return ModelInput(
        courses=courses,
        shifts=shifts
    )