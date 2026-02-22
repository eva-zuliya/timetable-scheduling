from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from datetime import datetime, timedelta


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


class CourseBatch(Course):
    batch_number: int
    valid_start_domain: Optional[list[int]] = None

    @property
    def id(self):
        return f"[{self.company}]-[{self.name}]-[{self.batch_number}]"


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
                current += timedelta(days=1)
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

    @property
    def week_groups(self) -> dict[int, list[int]]:
        """
        Returns:
            week_index -> list of day indices (calendar index)
        Week starts on Monday.
        """

        week_groups = defaultdict(list)

        # Convert date strings back to datetime
        date_objs = [
            datetime.strptime(d.date, "%Y-%m-%d")
            for d in self.dates
        ]

        # Anchor to first Monday
        first_date = date_objs[0]
        first_monday = first_date - timedelta(days=first_date.weekday())

        for idx, dt in enumerate(date_objs):
            delta_days = (dt - first_monday).days
            week_idx = delta_days // 7
            week_groups[week_idx].append(idx)

        return dict(week_groups)


class ModelInput(BaseModel):
    calendar: Calendar
    venues: dict[str, Venue]
    trainers: dict[str, Trainer]
    courses: dict[str, CourseBatch]
    groups: dict[str, Group]