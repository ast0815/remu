# Agent Instructions

This file contains instructions and guidelines for agents working on this project.

## Pre-commit Checks

**IMPORTANT**: Always run the pre-commit checks before committing anything. This ensures code quality and consistency.

```bash
pre-commit run --all-files
```

If any checks fail, fix the issues and run the checks again until all pass.

## Testing

Run the test suite before making changes:

```bash
./run_tests.sh
```

## Documentation

Build the documentation to ensure it compiles correctly:

```bash
cd docs
make html
```

## Version Requirements

The project uses specific version constraints to ensure compatibility:
- NumPy: >= 1.21.0, < 3.0.0
- SciPy: >= 1.7.0, < 2.0.0
- PyYAML: >= 5.4
- Matplotlib: >= 3.5.0

These constraints prevent breaking changes while allowing for updates within compatible versions.