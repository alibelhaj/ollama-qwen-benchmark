FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir requests==2.31.0

COPY llm_bench.py demo.py ablation_format.py concurrency.py ./

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3"]
CMD ["demo.py", "--exemples"]
