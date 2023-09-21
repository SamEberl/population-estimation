# Use an official PyTorch base image
FROM pytorch/pytorch:latest

RUN apt-get -y update

# Install additional dependencies (e.g., if you need other libraries)
RUN pip install numpy
RUN pip install matplotlib
RUN pip install tqdm
RUN pip install rasterio
RUN pip install pandas
RUN pip install tensorboard
RUN pip install albumentations
RUN pip install tensorboardX
RUN pip install tifffile
RUN pip install timm

# Copy your training code and data into the container
COPY /home/sameberl/population-estimation /home/sameberl/population-estimation

# Specify the command to run when the container starts
#CMD ["python", "train.py"]