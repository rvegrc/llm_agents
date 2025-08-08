FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY .env .
COPY langgraph_agent.py .
COPY /tools ./tools

CMD ["python", "api.py"]
