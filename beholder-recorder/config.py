"""Class-based Flask app configuration."""
from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '/srv/beholder_config/.env'))


class Config:
    """Configuration from environment variables."""

    SECRET_KEY = environ.get('SECRET_KEY')
    FLASK_APP = 'wsgi.py'

    # Static Assets
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'

    # Database
    SQLALCHEMY_DATABASE_URI = environ.get("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Security
    SECURITY_PASSWORD_SALT = environ.get('SECRET_KEY')
    SECURITY_RECOVERABLE = False

    CONFIG = "/home/beholder/beholder.ini"
