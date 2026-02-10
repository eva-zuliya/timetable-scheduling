from pydantic import BaseModel


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