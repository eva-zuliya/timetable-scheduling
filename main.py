from solver import run_solver


params = {

    'report_name': 'report',

    'file_master_venue': 'data_source/[Data] Master Training Scheduling - PAS - Master Venue.csv',
    'file_master_trainer': 'data_source/[Data] Master Training Scheduling - PAS - Master Trainer.csv',
    'file_master_course': 'data_source/[Data] Master Training Scheduling - PAS - Master Course.csv',
    'file_master_trainee': 'data_source/[Data] Master Training Scheduling - PAS - Master Employee.csv',
    'file_master_course_trainer': 'data_source/[Data] Master Training Scheduling - PAS - Master Course Trainer.csv',
    'file_master_course_sequence': 'data_source/[Data] Master Training Scheduling - PAS - Master Course Sequence.csv',
    'file_master_course_trainee': 'data_source/[Data] Master Training Scheduling - PAS - Master Course Employee.csv',

    'minimum_course_participant': 30,
    'maximum_group_size': 2,

    'days': 19,
    'hours_per_day': 8,
    'maximum_session_length': 8,

    'max_time_in_seconds': 900,
    'num_search_workers': 8,

    'is_considering_shift': False
}


if __name__ == "__main__":
    run_solver(params)



