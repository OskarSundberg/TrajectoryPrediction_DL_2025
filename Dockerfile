
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Install necessary packages
RUN apt update && apt install -y python3-all python3-pip ffmpeg libsm6 libxext6

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy code into the container
COPY ./code /code

# Set the working directory
WORKDIR /code

# Copy requirements file
COPY ./code/requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /code/requirements.txt

# Set the command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]