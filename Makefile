.DEFAULT_GOAL := build
.PHONY: build

build: configure build_webapp build_recorder


OUT_PATH=$(shell awk -F '=' '/^out_path/ {print $$2}' /home/${USER}/Desktop/beholder.ini)

DB=$(shell awk -F '=' '/^database/ {print $$2}' /home/${USER}/Desktop/beholder.ini)
KEY=$(shell openssl rand -base64 12)
HASH=$(shell openssl rand -base64 12)
SALT=$(shell openssl rand -base64 12)
define ENV_BODY
FLASK_APP=wsgi.py
SECRET_KEY=$(KEY)

SECURITY_PASSWORD_HASH=$(HASH)
SECURITY_PASSWORD_SALT=$(SALT)

SQLALCHEMY_DATABASE_URI=sqlite:///$(DB)
endef
export ENV_BODY
configure:
	# create webapp environment file
	@echo "$$ENV_BODY" > /srv/beholder-config/.env
	mkdir -p $(OUT_PATH)

build_recorder:
	cd beholder-recorder && make --makefile=Makefile.recorder build

build_webapp:
	cd beholder-recorder && make --makefile=Makefile.webapp build

clean:
	cd beholder-recorder && make --makefile=Makefile.recorder clean
	cd beholder-recorder && make --makefile=Makefile.webapp clean
