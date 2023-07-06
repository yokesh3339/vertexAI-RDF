FROM tiangolo/uvicorn-gunicorn:python3.8-slim
ADD requirements.txt .
# RUN pip install Cython --install-option="--no-cython-compile"
# RUN pip install scikit-learn==0.21.3
RUN pip install -r requirements.txt
COPY . .
# COPY requirements.txt requirements.txt
# RUN pip install Cython --install-option="--no-cython-compile"

ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
# COPY ./sentiment /sentiment