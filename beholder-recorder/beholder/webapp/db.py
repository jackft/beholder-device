import click  # type: ignore
from flask.cli import with_appcontext  # type: ignore
from flask_sqlalchemy import SQLAlchemy  # type: ignore

db = SQLAlchemy()

def init_db():
    from . import models  # noqa: F401
    db.create_all()

def close_db(e=None):
    pass

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    db.init_app(app)
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
