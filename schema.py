from pydantic import BaseModel
from typing import Literal, Optional


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
    file_master_course_batch: Optional[str] = None

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
    is_splitting_batch: bool = False

    course_stream: Optional[list[str]] = None
    companies: Optional[list[str]] = None

    max_time_in_seconds: int = 100
    num_search_workers: int = 8