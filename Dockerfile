FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve_api_class.py .
COPY flight_price_model.pkl .
EXPOSE 8000
CMD ["python", "serve_api_class.py"]
