.PHONY: format lint

format:
	uv run black .

lint: format
	uv run pylint app
