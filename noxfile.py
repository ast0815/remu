import nox

nox.needs_version = ">=2024.3.2"
# Use uv for the session virtualenvs when available (fast), fall back to
# virtualenv otherwise (e.g. in environments without uv).
nox.options.default_venv_backend = "uv|virtualenv"

# The example scripts use multiprocess/dill, and dill 0.4.1 (the current
# release) still reads code.co_lnotab, which Python 3.15 removed - so the
# examples run up to 3.14 until dill releases its fix (uqfoundation/dill#755).
# The test sessions below still cover 3.15.


@nox.session(python=["3.11", "3.12", "3.13", "3.14", "3.15"])
def tests(session):
    # Editable so that --cov=src measures the sources the tests import.
    session.install("-e", ".[mcmc,test]")
    session.run(
        "pytest",
        "--cov=src",
        "--cov-report=term-missing",
        "-Wdefault::DeprecationWarning:remu",
        "-Wignore::DeprecationWarning",
        "tests.py",
    )


@nox.session(python=["3.11", "3.14"])
def examples(session):
    session.install("-e", ".[mcmc,test,examples]")
    session.run("./run_examples.sh")


@nox.session(python="3.12")
def lint(session):
    session.install(
        "-e", ".[mcmc,test]", "flake8", "black", "isort", "mypy", "types-PyYAML"
    )
    session.run("flake8", "src/", "tests.py")
    session.run("black", "--check", "src/", "tests.py")
    session.run("isort", "--profile", "black", "--check", "src/", "tests.py")
    session.run("mypy", "src/")
