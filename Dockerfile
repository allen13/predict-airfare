FROM ghcr.io/allen13/conda-base-image:4.9.2

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY ./ /app
WORKDIR "/app"

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]