from .schema import Group
import pandas as pd


def week_to_horizon_slots(
    week_groups: dict[int, list[int]],
    week_shifts: dict[int, int],   # {week_index: shift}
    max_day_working_hours: int
):

    # ✅ If all weeks are NonShift (0), no restriction needed
    if all(shift == 0 for shift in week_shifts.values()):
        return None

    valid_slots = []
    half = max_day_working_hours // 2

    for week_idx, shift in week_shifts.items():

        if week_idx not in week_groups:
            continue

        for day_index in week_groups[week_idx]:

            day_start = day_index * max_day_working_hours

            # NonShift → full day
            if shift == 0:
                valid_slots.extend(
                    range(day_start,
                          day_start + max_day_working_hours)
                )

            # Overlap Shift1 → use second half
            elif shift == 1:
                valid_slots.extend(
                    range(day_start + half,
                          day_start + max_day_working_hours)
                )

            # Overlap Shift2 → use first half
            elif shift == 2:
                valid_slots.extend(
                    range(day_start,
                          day_start + half)
                )

            # Shift3 → no slots
            elif shift == 3:
                continue

    return valid_slots


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
