from invoke import task

@task
def hello(c):
    """Prints hello world"""
    c.run("echo 'Hello, world!'")

@task
def build(ctx):
    """_summary_

    Parameters
    ----------
    ctx : _type_
        _description_
    """

    ctx.run("docker build -t nickosipov/production-code-lesson:latest .")
