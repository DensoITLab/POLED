# Extend from dockerbase image
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV no_proxy="10.81.0.0/16,172.16.0.0/16,localhost"
ENV http_proxy="http://10.81.247.64:8080"
ENV https_proxy="http://10.81.247.64:8080"

# Install HDF5 library from ROOT user
RUN apt-get update && apt-get install -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    git \
    vim \
    wget \
    curl \
    htop \
    g++ \
    libhdf5-dev \
    ffmpeg \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash user

# Ensure the user user has permissions for /fiftyone
RUN mkdir -p /fiftyone \
    && chown -R user:user /fiftyone

# Add /home/user/.local/bin to path
ENV PATH="/home/user/.local/bin:${PATH}"

# Create the examples directory and set permissions
RUN mkdir -p /home/user/app/examples \
    && chown -R user:user /home/user/app

# Set the home directory and switch to the non-root user
USER user
WORKDIR /home/user/app

# Install jupyter, torch, and torchvision as non-root user
COPY requirements.txt ./
RUN pip --no-cache-dir install -r requirements.txt

# Set the default command to run Jupyter Notebook from the app folder
CMD ["/bin/bash"]