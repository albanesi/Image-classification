## Dockerfile
FROM python:3.13.0
WORKDIR /usr/src/app
COPY app.py *.onnx labels_map.txt requirements.txt ./
COPY web web
RUN apt-get update && apt-get install -y libgl1
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]