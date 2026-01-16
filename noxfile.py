import nox


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def tests(session):
    session.install("-r", "requirements.txt")
    session.install("-r", "test-requirements.txt")
    session.install("-e", ".")
    session.run("coverage", "run", "--source='remu'", "tests.py")
    session.run("coverage", "report")


@nox.session(python=["3.8", "3.12"])
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
