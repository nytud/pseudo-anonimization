FROM python:3.10-buster


ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1
ENV JAVA_HOME=/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64
ENV PATH="$JAVA_HOME/bin:${PATH}"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server/
ENV EMTSV_NUM_PROCESSES=2


WORKDIR /app

COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir uwsgi cython numpy && python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir https://huggingface.co/huspacy/hu_core_news_trf/resolve/main/hu_core_news_trf-any-py3-none-any.whl
COPY . /app
EXPOSE 8000
CMD [“python”, “./main.py”]