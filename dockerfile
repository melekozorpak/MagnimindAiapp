# Step 1: Use an official Python runtime as a parent image
FROM python:3.12.3

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Install system dependencies for PDF processing and OCR
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libsm6 libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file to the container
COPY requirements.txt ./requirements.txt

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of the application code to the container
COPY . .

# Step 7: Expose the Streamlit port
EXPOSE 8501

# Step 8: Ensure `emoji.png` is included in the container (if necessary)
COPY emoji.png /app/

# Step 9: Command to run the Streamlit application
CMD ["streamlit", "run", "magnimind_ai_bot.py", "--server.port=8501", "--server.enableCORS=false"]
