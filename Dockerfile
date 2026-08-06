FROM python:3.12
ARG NODE_VERSION=24

RUN useradd -ms /bin/bash optimus
WORKDIR /home/optimus
USER optimus

# install nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.6/install.sh | bash
# set env
ENV NVM_DIR=/home/optimus/.nvm
# install node
RUN bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION"

COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install -r requirements.txt
COPY ./src ./src
COPY ./optimus.py ./optimus.py
COPY ./mcp.json ~/.optimus/mcp.json

# set ENTRYPOINT for reloading nvm-environment
ENTRYPOINT ["bash", "-c", "source $NVM_DIR/nvm.sh && exec \"$@\"", "--"]
CMD ["sleep", "inf"]