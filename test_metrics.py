import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def base_df():
    """Six employees across three departments, split evenly on overtime and satisfaction."""
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "overtime": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "monthly_income": [4000, 5000, 6000, 7000, 8000, 9000],
            "job_satisfaction": [1, 2, 1, 2, 1, 2],
            "attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------


def test_attrition_rate_fifty_percent():
    df = pd.DataFrame(
        {"employee_id": [1, 2, 3, 4], "attrition": ["Yes", "No", "No", "Yes"]}
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_zero():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_one_hundred():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_rounds_to_two_decimals():
    # 1 leaver out of 3 = 33.333...% → should round to 33.33
    df = pd.DataFrame(
        {"employee_id": [1, 2, 3], "attrition": ["Yes", "No", "No"]}
    )
    assert attrition_rate(df) == 33.33


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------


def test_attrition_by_department_columns(base_df):
    result = attrition_by_department(base_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(base_df):
    # Each department has 2 employees and 1 leaver → 50% each
    result = attrition_by_department(base_df)
    rates = result.set_index("department")["attrition_rate"]
    assert rates["Sales"] == 50.0
    assert rates["HR"] == 50.0
    assert rates["IT"] == 50.0


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["HR", "HR", "Sales", "Sales"],
            "attrition": ["Yes", "Yes", "Yes", "No"],  # HR=100%, Sales=50%
        }
    )
    result = attrition_by_department(df)
    assert list(result["department"]) == ["HR", "Sales"]


def test_attrition_by_department_leaver_counts(base_df):
    result = attrition_by_department(base_df)
    counts = result.set_index("department")["leavers"]
    assert counts["Sales"] == 1
    assert counts["HR"] == 1
    assert counts["IT"] == 1


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------


def test_attrition_by_overtime_columns(base_df):
    result = attrition_by_overtime(base_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "Yes", "No", "No"],  # overtime=100%, no overtime=0%
        }
    )
    result = attrition_by_overtime(df)
    rates = result.set_index("overtime")["attrition_rate"]
    assert rates["Yes"] == 100.0
    assert rates["No"] == 0.0


def test_attrition_by_overtime_partial_rate():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "Yes", "No"],
            "attrition": ["Yes", "No", "No", "No"],  # overtime=33.33%, no overtime=0%
        }
    )
    result = attrition_by_overtime(df)
    rates = result.set_index("overtime")["attrition_rate"]
    assert rates["Yes"] == 33.33


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------


def test_average_income_by_attrition_columns():
    df = pd.DataFrame(
        {"attrition": ["Yes", "No"], "monthly_income": [3000, 6000]}
    )
    result = average_income_by_attrition(df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values():
    df = pd.DataFrame(
        {
            "attrition": ["Yes", "Yes", "No", "No"],
            "monthly_income": [3000, 5000, 7000, 9000],
        }
    )
    result = average_income_by_attrition(df)
    yes_avg = result.loc[result["attrition"] == "Yes", "avg_monthly_income"].iloc[0]
    no_avg = result.loc[result["attrition"] == "No", "avg_monthly_income"].iloc[0]
    assert yes_avg == 4000.0   # (3000 + 5000) / 2
    assert no_avg == 8000.0    # (7000 + 9000) / 2


def test_average_income_by_attrition_rounds_to_two_decimals():
    df = pd.DataFrame(
        {
            "attrition": ["Yes", "Yes", "Yes"],
            "monthly_income": [1000, 2000, 3000],  # mean = 2000.0 exactly
        }
    )
    result = average_income_by_attrition(df)
    assert result.loc[result["attrition"] == "Yes", "avg_monthly_income"].iloc[0] == 2000.0


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------


def test_satisfaction_summary_columns(base_df):
    result = satisfaction_summary(base_df)
    assert list(result.columns) == [
        "job_satisfaction",
        "total_employees",
        "leavers",
        "attrition_rate",
    ]


def test_satisfaction_summary_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "job_satisfaction": [1, 1, 1, 2, 2, 2],
            "attrition": ["Yes", "No", "No", "Yes", "Yes", "No"],
        }
    )
    result = satisfaction_summary(df).set_index("job_satisfaction")
    # Group 1: 1 leaver / 3 employees = 33.33%
    # Group 2: 2 leavers / 3 employees = 66.67%
    assert result.loc[1, "attrition_rate"] == 33.33
    assert result.loc[2, "attrition_rate"] == 66.67


def test_satisfaction_summary_sorted_ascending():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [3, 3, 1, 1],
            "attrition": ["Yes", "No", "Yes", "No"],
        }
    )
    result = satisfaction_summary(df)
    assert list(result["job_satisfaction"]) == [1, 3]


def test_satisfaction_summary_uses_group_headcount_not_total_leavers():
    # This guards against the original bug where the denominator was total
    # company leavers instead of employees within each satisfaction group.
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "job_satisfaction": [1, 1, 1, 1, 2],
            "attrition": ["Yes", "No", "No", "No", "Yes"],
        }
    )
    result = satisfaction_summary(df).set_index("job_satisfaction")
    # Group 1: 1/4 = 25%, Group 2: 1/1 = 100%
    # With the old bug (denominator = 2 total leavers): Group 1 = 50%, Group 2 = 50%
    assert result.loc[1, "attrition_rate"] == 25.0
    assert result.loc[2, "attrition_rate"] == 100.0
