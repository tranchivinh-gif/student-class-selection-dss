FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src_real ./src_real
COPY artifacts/src_real ./artifacts/src_real
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src_real/web.py", "--server.address=0.0.0.0", "--server.port=8501"]
