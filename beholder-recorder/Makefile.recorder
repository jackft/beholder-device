SOURCE_PATH=beholder/recorder

.DEFAULT_GOAL := build

include ../common.mk

build: install

install: logs

logs:
	mkdir -p logs

clean: clean_data clean_logs

clean_data:
	rm -rf beholder-data

clean_logs:
	rm -rf logs

clean_run: clean_data clean_logs
