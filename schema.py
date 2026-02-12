from pydantic import BaseModel
from typing import Literal
import pandas as pd


class Venue(BaseModel):
    name: str
    capacity: int


class Trainer(BaseModel):
    name: str
    eligible: list[str]


class Course(BaseModel):
    name: str
    duration: int
    prerequisites: list[str]


class Trainee(BaseModel):
    name: str
    shift: Literal["Non Shift", "Shift 1", "Shift 2", "Shift 3", "NS"]
    courses: list[str]

    @property
    def shift_start_hour(self):
        if self.shift in ["Shift 2", "Shift 3"]:
            return 4
        
        return 0
    
    @property
    def shift_end_hour(self):
        if self.shift in ["Shift 2", "Shift 3"]:
            return 8
        
        elif self.shift in ["Shift 1"]:
            return 4
        
        return 0


class Group(BaseModel):
    name: str
    courses: list[str]
    trainees: list[str]
    subgroup: dict[str, list[str]] = None
    shift: Literal["Non Shift", "Shift 1", "Shift 2", "Shift 3", "NS"]
    shift_start_hour: Literal[0, 4]
    shift_end_hour: Literal[0, 4, 8]

    def split_subgroups(self, max_size: int):
        self.subgroup = {}
        for i in range(0, len(self.trainees), max_size):
            self.subgroup[f"U{i//max_size + 1}"] = self.trainees[i:i+max_size]