from typing import Literal
from pydantic import BaseModel
import math


class CourseStats(BaseModel):
    company: str
    name: str
    trainees: list[str]
    count_trainee: int
    count_trainers: int
    max_venue_capacity_available: int
    
    @property
    def max_batches(self):
        if self.count_trainers <= self.max_venue_capacity_available:
            return 3
        
        trainee_per_trainer = math.ceil(self.count_trainee/self.count_trainers)
        if trainee_per_trainer <= self.max_venue_capacity_available:
            return self.count_trainers + 3

        batch_per_trainer = math.ceil(trainee_per_trainer/self.max_venue_capacity_available)
        return self.count_trainers * batch_per_trainer


class TraineeShift(BaseModel):
    name: str
    week1: Literal[0,1,2,3] = 0
    week2: Literal[0,1,2,3] = 0
    week3: Literal[0,1,2,3] = 0
    week4: Literal[0,1,2,3] = 0

    @property
    def rotating_shift(self):
        return {
            0: self.week1,
            1: self.week2,
            2: self.week3,
            3: self.week4
        }


class ModelInput(BaseModel):
    courses: dict[str, CourseStats]
    shifts: dict[str, TraineeShift]
