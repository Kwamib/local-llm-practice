"""The same question, three times at temperature 0 and three times at 1."""

from ask import ask

QUESTION = "Suggest a name for a new coffee shop in a small town. Reply with the name only."

for temperature in (0, 1):
    print(f"temperature {temperature}:")
    for _ in range(3):
        print("  ", ask(QUESTION, temperature=temperature))
