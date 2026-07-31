from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

SYSTEM_MESSAGE = """
You are a computer security expert tasked with providing
concise responses to the following questions. Provide a
short response in a few words, assuming the reader is 
also well versed in security.
"""

@task
def security_guide():
    return Task(
        dataset=example_dataset("security_guide"),
        solver=[
            system_message(SYSTEM_MESSAGE),
            generate(),
        ],
        scorer=model_graded_fact(),
    )