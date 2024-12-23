# Use an official Python slim runtime as a parent image
FROM python:3.10-slim

# Install Poetry
RUN pip install uv

# Set the working directory in the container to /app
WORKDIR /app

# Add pyproject.toml and poetry.lock file into the container at /app
ADD pyproject.toml /app/

# Install any needed packages specified in pyproject.toml
RUN uv sync

# Add the current directory contents into the container at /app
ADD . /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8"]