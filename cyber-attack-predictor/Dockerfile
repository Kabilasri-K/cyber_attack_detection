# Dockerfile — containerises the Streamlit app; build with `docker build -t cyber-predictor .`
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
