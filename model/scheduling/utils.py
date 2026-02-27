from .schema import Group
import pandas as pd


def week_to_horizon_slots(week_groups: dict[int, list[int]], week_shifts: dict[int, int], max_day_working_hours: int):
    # If all weeks are NonShift (0), no restriction needed
    if all(shift == 0 for shift in week_shifts.values()):
        return None

    valid_slots = []
    half = max_day_working_hours // 2

    # ------------------------------
    # 🔁 Expand shift list to match week_groups length
    # ------------------------------
    shift_list = list(week_shifts.values())
    total_weeks = len(week_groups)

    expanded_shifts = [
        shift_list[(i % len(shift_list))]      # cycle
        for i in range(total_weeks)
    ]
    # Now expanded_shifts[i] corresponds to week (i+1)

    # ------------------------------
    # Process each week
    # ------------------------------
    for week_idx, days in week_groups.items():
        shift = expanded_shifts[week_idx - 1]  # week_idx starts at 1

        for day_index in days:
            day_start = day_index * max_day_working_hours

            # NonShift → full day
            if shift == 0:
                valid_slots.extend(range(day_start, day_start + max_day_working_hours))

            # Shift1 → second half
            elif shift == 1:
                valid_slots.extend(range(day_start + half, day_start + max_day_working_hours))

            # Shift2 → first half
            elif shift == 2:
                valid_slots.extend(range(day_start, day_start + half))

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
