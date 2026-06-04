FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir "rq>=2.0" "redis>=5" rq-dashboard

EXPOSE 9181

CMD ["rq-dashboard", "--redis-url", "redis://redis:6379/0", "--bind", "0.0.0.0", "--port", "9181"]
