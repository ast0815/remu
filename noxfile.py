import nox

# The example scripts use multiprocess/dill, and dill 0.4.1 (the current
# release) still reads code.co_lnotab, which Python 3.15 removed - so the
# examples run up to 3.14 until dill releases its fix (uqfoundation/dill#755).
# The test sessions below still cover 3.15.


@nox.session(python=["3.11", "3.12", "3.13", "3.14", "3.15"])
def tests(session):
    session.install("-r", "requirements.txt")
    session.install("-r", "test-requirements.txt")
    session.install("-e", ".")
    session.install("pytest", "pytest-cov")
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
    session.install("-r", "requirements.txt")
    session.install("-r", "test-requirements.txt")
    session.install("-r", "example-requirements.txt")
    session.install("-e", ".")
    session.run("./run_examples.sh")


@nox.session(python="3.12")
def lint(session):
    session.install("-r", "test-requirements.txt")
    session.run("flake8", "src/", "tests.py")
    session.run("black", "--check", "src/", "tests.py")
    session.run("isort", "--check", "src/", "tests.py")
    session.run("mypy", "src/")
