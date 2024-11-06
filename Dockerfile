FROM ubuntu:23.10
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ARG DATAFOG_DEPLOYMENT_TYPE
ENV DATAFOG_DEPLOYMENT_TYPE=$DATAFOG_DEPLOYMENT_TYPE
ARG DATAFOG_API_VERSION
ENV DATAFOG_API_VERSION=$DATAFOG_API_VERSION

EXPOSE 8000


RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    vim \
    git \
    python3.11-dev \
    python3-dev \
    python3.11-venv \
    wget \
    poppler-utils

RUN useradd -ms /bin/bash datafoguser

ADD app /home/datafoguser/app

# Create and set permissions for temp directories
RUN mkdir -p /tmp /home/datafoguser/tmp && \
    chmod 777 /tmp && \
    chmod 777 /home/datafoguser/tmp && \
    chown -R datafoguser:datafoguser /home/datafoguser

RUN python3.11 -m venv .venv && . .venv/bin/activate && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r /home/datafoguser/app/requirements.txt

WORKDIR /home/datafoguser/app
USER datafoguser
ENTRYPOINT ["sh", "docker-entrypoint.sh"]