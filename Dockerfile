FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim
COPY . /
RUN pip install --no-cache-dir requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
# COPY ./sentiment /sentiment