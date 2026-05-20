# Base Image
FROM python:3.12-slim

# Set Working Directory
WORKDIR /app

# Copy Files
COPY . .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit Port
EXPOSE 8501

# Streamlit Config
ENV PYTHONUNBUFFERED=1

# Start Streamlit App
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]