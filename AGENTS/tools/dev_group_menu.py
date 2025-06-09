import sys

CODEBASES = {
    "speaktome": ["plot", "ml", "dev", "ctensor", "jax", "numpy"],
    "time_sync": [],
    # Add other codebases and their groups here
}

def ask(prompt, timeout=3, default="n"):
    import signal
    def handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        resp = input(prompt)
        signal.alarm(0)
        return resp.strip().lower() or default
    except TimeoutError:
        print()
        return default

selected_codebases = []
for cb in CODEBASES:
    if ask(f"Work on codebase '{cb}'? [y/N] (auto-skip in 3s): ") == "y":
        selected_codebases.append(cb)

selected_groups = {}
for cb in selected_codebases:
    groups = CODEBASES[cb]
    selected_groups[cb] = []
    for group in groups:
        if ask(f"Install group '{group}' for '{cb}'? [y/N] (auto-skip in 3s): ") == "y":
            selected_groups[cb].append(group)

print("Selected codebases:", selected_codebases)
print("Selected groups:", selected_groups)
# Optionally: write to a file or export as needed