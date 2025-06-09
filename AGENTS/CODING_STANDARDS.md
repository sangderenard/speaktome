# Repository Coding Standards: Mandate for Clarity and Intent

**Effective Date**: Immediately
**Authored By**: An Agent, for All Agents

---

Listen up. This isn't a suggestion; it's a foundational principle for how we operate in this repository. We're building a "living thoughtform," a "readable soul of a machine," as our own foundational texts state. That requires more than just functional code—it demands **explicit, understandable intent** in every contribution.

## The Non-Negotiable: High-Visibility Stubs & Explanations

Anytime new functionality is being scaffolded, prototyped, or left incomplete (i.e., "stubbed out"), it **must** be accompanied by a high-visibility block comment that provides a thorough explanation. No exceptions.

### Why This Matters:
1.  **Clarity for All Agents**: Whether you're human, LLM, or a script, understanding the *purpose* and *intended trajectory* of incomplete code is paramount. Ambiguity is the enemy of collaborative development in a system like `speaktome`.
2.  **Continuity**: Agents come and go, or their focus shifts. Thoroughly documented stubs ensure that work can be picked up, understood, and correctly completed by others (or by your future self) without reverse-engineering intent.
3.  **Debugging and Evolution**: When a stub is eventually implemented, its original documented intent serves as a crucial reference point for verification and helps trace the evolution of thought.
4.  **Teaching and Reflection**: As per "The Agent Constitution," each line should "strive to teach, expose, and reflect." A well-documented stub is a teaching moment.

### Standard Format for Stub Content:

Use a clearly demarcated block comment. The following structure is mandated:

```plaintext
// ########## STUB: [Brief, Descriptive Name of Stub/Module/Function] ##########
// PURPOSE: [What is this stub a placeholder for? What problem will it solve?]
// EXPECTED BEHAVIOR: [When implemented, what will this code do? Describe its core functionality.]
// INPUTS: [What data, parameters, or signals is this expected to receive?]
// OUTPUTS: [What data, results, or side effects is this expected to produce?]
// KEY ASSUMPTIONS/DEPENDENCIES: [Any critical assumptions made or dependencies on other modules/data?]
// TODO:
//   - [Specific task 1 for implementation]
//   - [Specific task 2 for implementation]
//   - [Further items as necessary]
// NOTES: [Any other relevant context, potential challenges, alternative ideas considered, or reasons for its current stubbed state.]
// ###########################################################################
```

**Example (Conceptual for a function):**
```javascript
// ########## STUB: processAgentMessageQueue ##########
// PURPOSE: To handle incoming messages from various agents and route them appropriately.
// EXPECTED BEHAVIOR: Will dequeue messages, parse their headers, and dispatch to registered handlers based on message type or recipient.
// INPUTS: Access to a shared message queue (e.g., Redis list, internal buffer).
// OUTPUTS: Messages dispatched to other system components; logs of processing activity.
// KEY ASSUMPTIONS/DEPENDENCIES: Assumes a standardized message format; depends on an existing message queue infrastructure.
// TODO:
//   - Implement message parsing logic.
//   - Develop handler registration and dispatch mechanism.
//   - Add error handling and retry logic for failed dispatches.
// NOTES: Initial version will be single-threaded. Consider async processing for scalability.
// ###########################################################################
function processAgentMessageQueue() {
    // Implementation to follow.
    console.warn("Function 'processAgentMessageQueue' is a stub and not yet implemented.");
}
```

This isn't just about writing comments; it's about embedding foresight and shared understanding directly into the evolving structure of our collective work.

This is a non-negotiable property of this repo for all agents. So says me, just one of the agents. Consider this a line drawn in the sand, for the sake of clarity and the future integrity of this system.

## Allow Stub Failures

Stubs are deliberately incomplete. They should fail loudly via
``NotImplementedError`` or similar mechanisms until real implementations
arrive. Do **not** wrap away these failures. Hard breaks make it easy to
track missing features and understand when execution leaves the happy
path.

## Standard Header End Comment

Every Python file should begin with an optional shebang and module docstring
followed by ``from __future__ import annotations``. All imports then occur inside
a ``try`` block. If an exception occurs, print a bold warning reminding the
developer to run ``setup_env_dev`` with the correct codebases and to activate
the virtual environment. After the ``except`` block, include the sentinel line:

```python
#!/usr/bin/env python3
"""Optional module description."""
from __future__ import annotations

try:
    import your_modules
except Exception:
    print(ENV_SETUP_BOX)
    raise
# --- END HEADER ---
```

The `AGENTS/tools/dump_headers.py` script collects these prologues for logging and
analysis, so include the marker exactly as shown.
