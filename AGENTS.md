# ReMU Agent Guidelines

## Updating the AGENTS.md file

- Whenever you laern something new about this project, update the `AGENTS.md` file
- When something seems contradictory, ask the user for clarification

## Environment

- All development and testing must happen in a virtual Python environment
- The environment directory is `ENV` at the base of the project
- The requirements for running tests, building the documentation are stored in the `*-requirements.txt` files

## Build/Lint/Test Commands

- Run all tests: `nox -s tests`
- Run all examples: `nox -s examples`
  - **Danger** These examples can take a long time to run! Run only when strictly necessary!
- Run single test: `nox -s tests -- -k test_name`
- Single examples must be run from within the `docs/example/<example_number>` directory.
- Lint code: `flake8 src/ tests.py`
- Format code: `black src/ tests.py` and `isort src/ tests.py`
- Type check: `mypy src/`
- Install dependencies: `pip install -r requirements.txt && pip install -r test-requirements.txt`

## Pre-commit Checks

**IMPORTANT**: Always run the pre-commit checks before committing anything. This ensures code quality and consistency.

```bash
pre-commit run --all-files
```

If any checks fail, fix the issues and run the checks again until all pass.

## Code Style Guidelines

### Imports
- Use `import numpy as np`, `import scipy as sp`, `import pandas as pd`
- Follow PEP 8 import conventions
- Group imports: standard library, third-party, local packages
- Use `from module import specific_function` for clarity

### Formatting
- Use Black for code formatting
- Use isort for import sorting
- Maintain 88 character line width
- Use 4-space indentation
- Add blank lines between top-level functions and classes

### Types
- Use type hints from typing module where appropriate
- Include type annotations for function parameters and return values
- Prefer `List[T]`, `Dict[K, V]` over `list`, `dict` for better clarity
- Use Union types for multiple possible types

### Naming Conventions
- Variables and functions: snake_case
- Classes: CamelCase
- Constants: UPPER_CASE
- Private methods: _private_method_name

### Error Handling
- Use standard Python exceptions (ValueError, TypeError, etc.)
- Avoid bare `except:` clauses
- Provide meaningful error messages
- Use try/except blocks for expected failure cases

### Testing
- Write unit tests for new functionality
- Use pytest-style test functions (test_* naming)
- Test edge cases and error conditions
- Maintain 100% test coverage when possible
- Use setUp methods for test fixtures

## Project Structure

- Source code: `src/remu/`
- Tests: `tests.py` (monolithic test file)
- Examples: `docs/examples/`
- Documentation: `docs/`

## Development Tools

- Pre-commit hooks configured via `.pre-commit-config.yaml`
- Linting with flake8
- Formatting with black
- Import sorting with isort
- Type checking with mypy

## Additional Notes

This project uses Python 3.8+ and follows PEP 8 style guide.
Code should be well-documented with docstrings.
Use meaningful variable and function names.
Avoid magic numbers and strings.
Follow the existing code patterns in the repository.
