"""Initialize Flask app."""
from flask import Flask
from flask_security import Security, SQLAlchemySessionUserDatastore  # type: ignore

def create_app():
    """Create Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')

    with app.app_context():
        from .db import db
        from .db import init_app as db_init_app
        db_init_app(app)
        from .models import User, Role
        user_datastore = SQLAlchemySessionUserDatastore(db.session, User, Role)
        Security(app, user_datastore)

        # Import parts of our application
        from .home import home  # type: ignore
        app.register_blueprint(home.home_bp)

        return app
