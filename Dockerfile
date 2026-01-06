# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies from requirements.txt
# Then install specific versions of langchain, huggingface, transformers, etc.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       langchain==0.3.27 \
       langchain-core==0.3.80 \
       langchain-community==0.3.31 \
       langchain-huggingface==0.3.1 \
       langchain-text-splitters==0.3.11 \
       huggingface-hub==0.36.0 \
       tokenizers==0.19.1 \
       transformers==4.43.3 \
       sentence-transformers==2.7.0 \
       requests==2.32.5

# Expose port if needed (Flask default 8000)
EXPOSE 8000

# Run the server
CMD ["python", "-u", "server.py"]