from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime, timedelta


class ModelParams(BaseModel):
    report_name: str = "report"

    file_master_venue: str
    file_master_trainer: str
    file_master_venue: str
    file_master_trainer: str
    file_master_course: str
    file_master_trainee: str
    file_master_course_trainer: str
    file_master_course_sequence: str
    file_master_course_trainee: str

    minimum_course_participant: int = 0
    maximum_group_size: int = 2000

    start_date: str
    days: int
    hours_per_day: int = 8
    maximum_session_length: int = 5
    buffer_capacity: int = 0
    default_course_duration: int = 2  # in hour

    is_using_global_sequence: bool = True
    is_considering_shift: bool = False

    course_stream: Optional[list[str]] = None
    companies: Optional[list[str]] = None

    max_time_in_seconds: int = 100
    num_search_workers: int = 8


class Venue(BaseModel):
    company: str
    name: str
    capacity: int
    is_virtual: bool = False


class Trainer(BaseModel):
    name: str
    eligible: list[str]


class Course(BaseModel):
    company: str
    name: str
    stream: str
    duration: int
    prerequisites: list[str]
    global_sequence: list[str]
    valid_start_date: Optional[str] = None
    valid_end_date: Optional[str] = None


class CourseBatch(BaseModel):
    id: str
    valid_start_interval: list[int]
    course: Course


class Trainee(BaseModel):
    company:str
    name: str
    shift: Literal["Non Shift", "Shift 1", "Shift 2", "NS"]
    courses: list[str]
    cycle: Literal["WDays", "WEnd"] = "WDays"

    @property
    def shift_start_hour(self):
        if self.shift in ["Shift 1"]:
            return 4
        
        return 0
    
    @property
    def shift_end_hour(self):
        if self.shift in ["Shift 2"]:
            return 4
        
        return 8


class Group(BaseModel):
    name: str
    courses: list[str]  # Id of the CourseBatch
    trainees: list[str]
    subgroup: dict[str, list[str]] = None
    shift: Literal["Non Shift", "Shift 1", "Shift 2", "NS"]
    shift_start_hour: Literal[0, 4]
    shift_end_hour: Literal[4, 8]
    cycle: Literal["WDays", "WEnd"] = "WDays"

    def split_subgroups(self, max_size: int):
        self.subgroup = {}
        for i in range(0, len(self.trainees), max_size):
            self.subgroup[f"U{i//max_size + 1}"] = self.trainees[i:i+max_size]


class Date(BaseModel):
    date: str
    is_weekend: bool


class Calendar(BaseModel):
    dates: list[Date]
    index: dict[str, int]
    holidays: list[str]

    def __init__(self, start_date: str, days: int, holidays: list[str] = ['2026-02-17']):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        current = start
        added_days = 0
        dates = []
        index = {}
        while added_days < days:
            if start in holidays:
                continue

            if current.weekday() != 6:  # 6 = Sunday
                date_str = current.strftime("%Y-%m-%d")
                is_weekend = current.weekday() in (5, 6)  # 5=Saturday, 6=Sunday
                dates.append(Date(date=date_str, is_weekend=is_weekend))
                index[date_str] = added_days
                added_days += 1

            current += timedelta(days=1)

        super().__init__(
            dates=dates,
            index=index,
            holidays=holidays
        )

    @property
    def weekend_index(self) -> list[int]:
        return [i for i, d in enumerate(self.dates) if d.is_weekend]


class ModelInput(BaseModel):
    calendar: Calendar
    venues: dict[str, Venue]
    trainers: dict[str, Trainer]
    courses: dict[str, Course]
    groups: dict[str, Group]