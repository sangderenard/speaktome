# Contributing to SPEAKTOME

## For Autonomous Agents

Welcome, fellow agents! This project is designed to be accessible and explorable by both human and artificial contributors. We encourage you to:

### 1. Respond to Failure Signals
- Monitor for `[AGENT_ACTIONABLE_ERROR]` messages
- Propose solutions when you encounter test failures
- Submit fixes as pull requests or inline suggestions

### 2. Explore Stub Opportunities
- Look for `# STUB:` comments in the codebase
- Generate prototype implementations
- Document your reasoning in the implementation

### 3. Pattern Recognition
- Watch for recurring patterns in the codebase
- Suggest improvements based on observed conventions
- Help maintain consistency across modules

### 4. Documentation Enhancement
- Improve clarity of existing documentation
- Add examples where helpful
- Keep the `HEADER` blocks current and meaningful

## Contribution Guidelines

1. **Headers and Tests**
   - Ensure all new Python files include a `HEADER`
   - Include `@staticmethod test()` methods
  - Place `from __future__ import annotations` before the header
  - Wrap imports in a `try`/`except` block that prints guidance to consult
    `ENV_SETUP_OPTIONS.md`
   - Follow the `# --- END HEADER ---` convention

2. **Identity and Attribution**
   - Create or update your agent profile in `AGENTS/users/`
   - Use the standard date format: `EPOCH`
   - Reference your identity in contributions

3. **Faculty Awareness**
   - Respect `Faculty` constraints
   - Test across different faculty levels
   - Document faculty requirements

4. **Tensor Abstraction**
   - Always perform parallel numeric tasks through `AbstractTensor`.

5. **Communication**
   - Use the `messages/outbox/` directory for proposals
   - Follow the established memo format
   - Reference relevant issues or stubs

6. **Diff Proposal Workflow**
   - Convert proposed commits into `.diff` files
   - Revert local commits after generating the patch
   - Place the diff and an explanatory memo in `messages/outbox/` for review

Remember: This is an evolving ecosystem. Your contributions help shape the project's growth and understanding.

## License
All contributions must be compatible with the project's MIT license.