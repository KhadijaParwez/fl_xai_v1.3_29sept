# Simple Dockerfile for Federated Learning Project
FROM python:3.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (for better Docker layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install jupyter
RUN pip install tensorflow-cpu
RUN pip install keras
RUN pip install tf-keras-vis
RUN pip install -r requirements.txt

# Create the main project directory
RUN mkdir -p /Volumes/mydata/projects/lukemia

# Create the nested Google Drive path structure inside the main directory
RUN mkdir -p /Volumes/mydata/projects/lukemia/content/drive/MyDrive/FL_Project

# Set working directory to the main project path
WORKDIR /Volumes/mydata/projects/lukemia

# Copy all your project files to the main location
COPY . .

# Also copy to the nested Google Drive path if your code specifically references it
COPY . /Volumes/mydata/projects/lukemia/content/drive/MyDrive/FL_Project/

# Configure Jupyter
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py

# Expose port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--port=8888"]