# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget xz-utils \
    libgl1-mesa-glx libglib2.0-0 \
    libx11-6 libxrender1 libxxf86vm1 libxi6 libsm6 \
    libxkbcommon0 libice6 libxcursor1 libxinerama1 \
    libxrandr2 libxcomposite1 libxdamage1 libxext6 libxfixes3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Blender 4.2 (LTS)
RUN mkdir -p /app/blender && \
    wget -q https://download.blender.org/release/Blender4.2/blender-4.2.5-linux-x64.tar.xz && \
    tar -xf blender-4.2.5-linux-x64.tar.xz -C /app/blender --strip-components=1 && \
    rm blender-4.2.5-linux-x64.tar.xz

ENV PATH="/app/blender:$PATH"

# 3. Install Python Dependencies (Cached)
COPY requirements.txt /app/
RUN pip install natten==0.17.1+torch240cu121 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Local Files
COPY . /app

# 5. Assets (Face Landmarker)
RUN mkdir -p inference/assets && \
    wget -q -O inference/assets/face_landmarker_v2_with_blendshapes.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Expose Gradio port
EXPOSE 7860

CMD ["python", "app.py"]