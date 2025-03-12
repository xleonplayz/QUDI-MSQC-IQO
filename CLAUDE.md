# QUDI-MSQC-IQO Development Guide

## Build & Test Commands
- Install: `pip install -e .`
- Run tests: `pytest tests/`
- Run single test: `pytest tests/test_scan_constraints.py -v`
- Run notebook tests: `bash tests/test.sh`
- Build docs: `make html`

## Code Style Guidelines
- Python 3.8-3.10 compatible
- Follow PEP 8 naming conventions: snake_case for functions/variables, CamelCase for classes
- Use typing annotations for function parameters and return values
- Error handling: Include meaningful error messages with assertions/exceptions
- Use proper docstrings with parameter descriptions
- Organize imports: stdlib first, then third-party, then local modules
- Hardware modules should implement abstract interfaces fully
- Use Mutex locks for thread-safe operations
- Include example config in module docstring for hardware modules
- Base class methods should be called in derived classes with super()