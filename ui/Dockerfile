FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy requirements from ui/ folder
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Since you build from inside ui/ folder, copy everything from current context
COPY . .

EXPOSE 8501

# FIXED: app.py is directly in /app (because you build from ui/ folder)
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--logger.level=info", \
     "--server.runOnSave=false"]
