# Base image
FROM --platform=linux/amd64 python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY paper_summarizer/ paper_summarizer/
COPY data/ data/
COPY Makefile Makefile

WORKDIR /
RUN make requirements

ENTRYPOINT ["python", "-u", "paper_summarizer/train_model.py"]