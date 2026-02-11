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
    shift: Literal["NonShift", "Shift1", "Shift2", "Shift3"] = "NonShift"
    courses: list[str]


class Group(BaseModel):
    name: str
    courses: list[str]
    trainees: list[str]
    subgroup: dict[str, list[str]] = None

    def split_subgroups(self, max_size: int):
        self.subgroup = {}
        for i in range(0, len(self.trainees), max_size):
            self.subgroup[f"U{i//max_size + 1}"] = self.trainees[i:i+max_size]


def export_groups_to_df(groups: list[Group]):
    rows = []

    for g in groups:
        # ensure subgroup is generated
        if g.subgroup is None:
            raise ValueError(f"Group {g.name} has no subgroup. Run split_subgroups() first.")

        for subgroup_name, members in g.subgroup.items():
            for member in members:
                rows.append({
                    "group_name": g.name,
                    # "course": course,
                    "subgroup_name": subgroup_name,
                    "trainee": member
                })

    df = pd.DataFrame(rows)
    df.to_csv("groups_trainee.csv", index=False)