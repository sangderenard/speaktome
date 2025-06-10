try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    # Standard library imports
    import json
    from typing import Dict, Callable, Optional, List, Tuple

    # Local application/library specific imports
    from .scorer import Scorer
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

class HumanScorerPolicyManager:
    DEFAULT_POLICY_FILE = "human_scorer_policy.json"

    def __init__(self, scorer: Scorer):  # Forward reference Scorer
        self.scorer = scorer
        self.available_scorers: Dict[str, Callable] = Scorer.get_available_scoring_functions()
        self.load_policy()  # Load saved policy if exists

    def load_policy(self, filepath: Optional[str] = None) -> bool:
        fpath = filepath or self.DEFAULT_POLICY_FILE
        try:
            with open(fpath, 'r') as f:
                loaded_policy = json.load(f)
            if isinstance(loaded_policy, dict) and \
               loaded_policy.get("scorer_name") in self.available_scorers and \
               isinstance(loaded_policy.get("width"), int) and \
               isinstance(loaded_policy.get("temperature"), float):
                self.default_policy = loaded_policy
                print(f"Loaded scorer policy from {fpath}: {self.default_policy}")
                return True
            else:
                print(f"Warning: Invalid policy data in {fpath}. Using defaults.")
        except FileNotFoundError:
            print(f"Policy file {fpath} not found. Using default policy.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {fpath}. Using default policy.")
        except Exception as e:
            print(f"Error loading policy from {fpath}: {e}. Using default policy.")
        return False

    def save_policy(self, filepath: Optional[str] = None) -> bool:
        fpath = filepath or self.DEFAULT_POLICY_FILE
        try:
            with open(fpath, 'w') as f:
                json.dump(self.default_policy, f, indent=4)
            print(f"Saved scorer policy to {fpath}: {self.default_policy}")
            return True
        except Exception as e:
            print(f"Error saving policy to {fpath}: {e}")
            return False

    def set_default_policy_interactive(self):
        print("\n--- Configure Default Scorer Policy ---")
        print("Current policy:", self.default_policy)
        print("Available scorers:")
        scorer_names = list(self.available_scorers.keys())
        for i, name in enumerate(scorer_names):
            print(f"  {i}: {name}")
        try:
            choice_idx = int(input(f"Choose scorer (0-{len(scorer_names)-1}): "))
            if not (0 <= choice_idx < len(scorer_names)):
                print("Invalid scorer choice.")
                return
            chosen_scorer_name = scorer_names[choice_idx]
            width = int(input(f"Enter bin width for '{chosen_scorer_name}' (integer, e.g., 5): "))
            if width <= 0:
                print("Width must be positive.")
                return
            temperature = float(input(f"Enter temperature for '{chosen_scorer_name}' (float, e.g., 1.5): "))
            if temperature <= 0:
                print("Temperature must be positive.")
                return
            self.default_policy = {"scorer_name": chosen_scorer_name, "width": width, "temperature": temperature}
            print("New policy set:", self.default_policy)
            if input("Save this policy as default? (y/n): ").lower() == 'y':
                self.save_policy()
        except ValueError:
            print("Invalid input type.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_current_score_bins(self) -> List[Tuple[Callable, int, float]]:
        scorer_fn = self.available_scorers.get(self.default_policy["scorer_name"])
        if not scorer_fn:
            print(f"Warning: Scorer '{self.default_policy['scorer_name']}' not found. Defaulting to mean_logprob.")
            scorer_fn = Scorer.mean_logprob_score
            self.default_policy["scorer_name"] = "mean_logprob"
        return [(scorer_fn, self.default_policy["width"], self.default_policy["temperature"])]
