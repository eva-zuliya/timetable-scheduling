from .schema import Group
import pandas as pd


def export_groups_trainee_to_df(groups: list[Group], report_name: str) -> pd.DataFrame:
    rows = []

    for g in groups:
        # ensure subgroup is generated
        if g.subgroup is None:
            raise ValueError(f"Group {g.name} has no subgroup. Run split_subgroups() first.")

        for subgroup_name, members in g.subgroup.items():
            for member in members:
                rows.append({
                    "group_name": g.name,
                    "subgroup_name": subgroup_name,
                    "trainee": member
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"export/{report_name}_groups_trainee.csv", index=False)

    return df


def export_groups_courses_to_df(groups: list[Group], report_name: str) -> pd.DataFrame:
    rows = []

    for g in groups:
        # ensure subgroup is generated
        if g.subgroup is None:
            raise ValueError(f"Group {g.name} has no subgroup. Run split_subgroups() first.")

        for course in g.courses:
            rows.append({
                "group_name": g.name,
                "course": course
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"export/{report_name}_groups_courses.csv", index=False)

    return df


def hour_index_to_time(hour_idx, is_start: bool):
    # base start 08:00
    hour = 8 + hour_idx

    # skip lunch break at 12:00
    if hour_idx > 3:
        hour += 1

    if not is_start and hour_idx == 4:
        hour -= 1

    return f"{hour:02d}:00"
