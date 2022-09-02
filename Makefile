.DEFAULT_GOAL := build
.PHONY: build

build: build_recorder

build_recorder:
	cd recorder && make build
