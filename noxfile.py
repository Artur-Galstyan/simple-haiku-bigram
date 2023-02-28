import nox


@nox.session(reuse_venv=True)
def tests(session):
    # install requirements.txt
    session.install("-r", "requirements.txt")
    # install this package
    session.install("-e", ".")
    session.install("pytest")
    session.run("pytest")


@nox.session(reuse_venv=True)
def lint(session):
    session.install("flake8")
    session.run("flake8", "src")
