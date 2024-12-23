.PHONY: check format sort

check:
	uv run ruff check

format:
	uv run black .

sort:
	uv run isort .