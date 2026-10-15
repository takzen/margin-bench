from margin_bench.core import Score, Task, TaskFamily
from margin_bench.grading.parse import extract_json


def grade_response(family: TaskFamily, task: Task, text: str, with_plot: bool = True) -> Score:
    """Grade a raw model response: parse the JSON answer, then grade it."""
    answer = extract_json(text)
    if answer is None:
        return Score.format_failure("no JSON object found in the response")
    return family.grade(task, answer, with_plot=with_plot)
