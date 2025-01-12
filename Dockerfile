FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]