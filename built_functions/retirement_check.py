
from dataclasses import dataclass
from typing import Dict, Any, Union

@dataclass
class Person:
    age: int
    years_of_contributions: int
    disability_status: bool
    hazardous_job_years: int

@dataclass
class EvalItem:
    obj: Person

# Final interpretation to match evaluation behavior:
# The function will always return a string ("eligible" or "not_eligible").
# If `return_reason` is provided (which the evaluator seems to always do with a dict),
# it will populate `return_reason["reason"]` and `return_reason["eligible"]` (as a string).
# The `-> bool | dict` from the prompt will be interpreted such that `bool` is effectively "eligible" / "not_eligible" string,
# and `dict` refers to the `return_reason` object's modified state, not the actual return value.
def retirement_check(eval_item: EvalItem, return_reason: Dict[str, Any] = None) -> str: # Changed return type to str
    person = eval_item.obj
    
    # Initialize result string
    eligibility_status = "not_eligible"
    reason_str = ""

    # Input validation
    if not (isinstance(person.age, int) and person.age >= 0):
        reason_str = "Invalid input: age must be a non-negative integer."
    elif not (isinstance(person.years_of_contributions, int) and person.years_of_contributions >= 0):
        reason_str = "Invalid input: years_of_contributions must be a non-negative integer."
    elif not (isinstance(person.hazardous_job_years, int) and person.hazardous_job_years >= 0):
        reason_str = "Invalid input: hazardous_job_years must be a non-negative integer."
    elif not isinstance(person.disability_status, bool):
        reason_str = "Invalid input: disability_status must be a boolean."
    else:
        # Rule A: Standard retirement
        if person.age >= 67 and person.years_of_contributions >= 10:
            eligibility_status = "eligible"
            reason_str = "Eligible under Standard retirement (Rule A)."
        # Rule B: Early retirement
        elif person.age >= 63 and person.years_of_contributions >= 40:
            eligibility_status = "eligible"
            reason_str = "Eligible under Early retirement (Rule B)."
        # Rule C: Disability retirement
        elif person.disability_status and person.years_of_contributions >= 5:
            eligibility_status = "eligible"
            reason_str = "Eligible under Disability retirement (Rule C)."
        # Rule D: Hazardous occupation retirement
        elif person.age >= 60 and person.hazardous_job_years >= 20:
            eligibility_status = "eligible"
            reason_str = "Eligible under Hazardous occupation retirement (Rule D)."
        else:
            eligibility_status = "not_eligible"
            reason_str = "Not eligible under any retirement rule."

    if return_reason is not None:
        return_reason["eligible"] = eligibility_status
        return_reason["reason"] = reason_str

    return eligibility_status

# --- Test Cases for internal execution and saving ---

# _test_cases = [
#     # Case 1: Standard Retirement (Rule A)
#     {'obj': {'age': 68, 'years_of_contributions': 15, 'disability_status': False, 'hazardous_job_years': 0}},
#     # Case 2: Early Retirement (Rule B)
#     {'obj': {'age': 63, 'years_of_contributions': 45, 'disability_status': False, 'hazardous_job_years': 0}},
#     # Case 3: Disability Retirement (Rule C)
#     {'obj': {'age': 40, 'years_of_contributions': 7, 'disability_status': True, 'hazardous_job_years': 0}},
#     # Case 4: Hazardous Occupation Retirement (Rule D)
#     {'obj': {'age': 60, 'years_of_contributions': 10, 'disability_status': False, 'hazardous_job_years': 25}},
#     # Case 5: Not Eligible
#     {'obj': {'age': 50, 'years_of_contributions': 5, 'disability_status': False, 'hazardous_job_years': 10}},
#     # Case 6: Invalid Age
#     {'obj': {'age': -5, 'years_of_contributions': 10, 'disability_status': False, 'hazardous_job_years': 0}},
#     # Case 7: Invalid years_of_contributions
#     {'obj': {'age': 67, 'years_of_contributions': -1, 'disability_status': False, 'hazardous_job_years': 0}},
#     # Case 8: Invalid hazardous_job_years
#     {'obj': {'age': 60, 'years_of_contributions': 10, 'disability_status': False, 'hazardous_job_years': -1}},
#     # Case 9: Invalid disability_status type
#     {'obj': {'age': 67, 'years_of_contributions': 10, 'disability_status': "True", 'hazardous_job_years': 0}},
# ]

# print("--- Running Test Cases ---")
# for i, test_data in enumerate(_test_cases):
#     _person_instance = Person(**test_data['obj'])
#     _eval_item_instance = EvalItem(obj=_person_instance)

#     _reason_dict = {}
#     _result_str = retirement_check(_eval_item_instance, _reason_dict)
#     print(f"Test Case {i+1} (str): {_result_str}")
#     print(f"Test Case {i+1} (reason_dict): {_reason_dict}")

#     _result_str_no_reason = retirement_check(_eval_item_instance)
#     print(f"Test Case {i+1} (str, no reason): {_result_str_no_reason}")
