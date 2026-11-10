FROM python:3.12
# ARG NODE_VERSION=24

# RUN useradd -ms /bin/bash optimus
# WORKDIR /home/optimus
# USER optimus

COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install -r requirements.txt
COPY ./src ./src
COPY ./cli.py ./cli.py
COPY ./backend.py ./backend.py
COPY ./index.html ./index.html
COPY ./test_optimus.py ./test_optimus.py

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "-u", "backend.py", "--file", "/data/tasks.json"]