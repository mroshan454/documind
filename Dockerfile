# Python 3.11 Base Image 
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first 
COPY requirements.txt /app/

# Install dependencies 
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the app code 
COPY . /app/ /app/

# Tell Docker the container will listen on port 8000
EXPOSE 8000 

# Run uvicorn when the container starts 
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]