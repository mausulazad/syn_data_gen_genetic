# Use the PyTorch base image with CUDA support
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="/opt/conda/lib/python3.10/site-packages:."

# Install system dependencies
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        python3-pip \
        gcc \
        wget git \
        curl htop \
        zip unzip \
        libgl1 libglib2.0-0 libpython3-dev \
        gnupg g++ libusb-1.0-0 libsm6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the build dependencies
RUN pip install --upgrade pip \
    && pip install setuptools==61.0 wheel build

# Install torch with the appropriate CUDA version
RUN pip install torch==2.2.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install torchvision with the matching CUDA version
RUN pip install torchvision==0.17.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install the project dependencies
RUN pip install \
    absl-py==2.1.0 \
    accelerate==1.0.0 \
    aiohappyeyeballs==2.4.0 \
    aiohttp==3.10.5 \
    aiosignal==1.3.1 \
    annotated-types==0.7.0 \
    anyio==4.4.0 \
    argon2-cffi==23.1.0 \
    argon2-cffi-bindings==21.2.0 \
    arrow==1.3.0 \
    asttokens==2.4.1 \
    async-lru==2.0.4 \
    async-timeout==4.0.3 \
    attrs==24.2.0 \
    av==13.0.0 \
    babel==2.16.0 \
    backoff==2.2.1 \
    beautifulsoup4==4.12.3 \
    bitsandbytes==0.41.3 \
    bleach==6.1.0 \
    boto3==1.35.15 \
    botocore==1.35.15 \
    cachetools==5.5.0 \
    certifi==2024.8.30 \
    cffi==1.17.1 \
    charset-normalizer==3.3.2 \
    click==8.1.7 \
    cloudpickle==3.0.0 \
    comm==0.2.2 \
    contourpy==1.3.0 \
    cycler==0.12.1 \
    dask==2023.7.1 \
    datasets==2.16.1 \
    debugpy==1.8.5 \
    decorator==5.1.1 \
    defusedxml==0.7.1 \
    dill==0.3.7 \
    distro==1.9.0 \
    einops==0.8.0 \
    einops-exts==0.0.4 \
    evaluate==0.4.3 \
    exceptiongroup==1.2.2 \
    executing==2.1.0 \
    fastapi==0.114.0 \
    fastjsonschema==2.20.0 \
    filelock==3.16.0 \
    fire==0.6.0 \
    fonttools==4.53.1 \
    fqdn==1.5.1 \
    frozenlist==1.4.1 \
    fsspec==2023.10.0 \
    ftfy==6.2.3 \
    fuzzywuzzy==0.18.0 \
    google-auth==2.34.0 \
    google-auth-oauthlib==1.2.1 \
    greenlet==3.1.1 \
    grpcio==1.66.1 \
    h11==0.14.0 \
    httpcore==1.0.5 \
    httpx==0.27.2 \
    huggingface-hub==0.25.0 \
    idna==3.8 \
    importlib_metadata==8.5.0 \
    ipykernel==6.26.0 \
    ipython==8.17.2 \
    ipywidgets==8.1.1 \
    isoduration==20.11.0 \
    jedi==0.19.1 \
    Jinja2==3.1.4 \
    jiter==0.6.1 \
    jmespath==1.0.1 \
    joblib==1.4.2 \
    json5==0.9.25 \
    jsonpatch==1.33 \
    jsonpointer==3.0.0 \
    jsonschema==4.23.0 \
    jsonschema-specifications==2023.12.1 \
    jupyter-events==0.10.0 \
    jupyter-lsp==2.2.5 \
    jupyter_client==8.6.2 \
    jupyter_core==5.7.2 \
    jupyter_server==2.14.2 \
    jupyter_server_terminals==0.5.3 \
    jupyterlab==4.2.0 \
    jupyterlab_pygments==0.3.0 \
    jupyterlab_server==2.27.3 \
    jupyterlab_widgets==3.0.13 \
    keras==2.15.0 \
    kiwisolver==1.4.7 \
    langchain==0.3.6 \
    langchain-core==0.3.18 \
    langchain-ollama==0.2.0 \
    langchain-openai==0.2.8 \
    langchain-text-splitters==0.3.1 \
    langsmith==0.1.138 \
    Levenshtein==0.26.1 \
    lightning==2.4.0 \
    lightning-cloud==0.5.70 \
    lightning-utilities==0.11.7 \
    lightning_sdk==0.1.17 \
    locket==1.0.0 \
    Markdown==3.7 \
    markdown-it-py==3.0.0 \
    MarkupSafe==2.1.5 \
    matplotlib==3.8.2 \
    matplotlib-inline==0.1.7 \
    mdurl==0.1.2 \
    mistune==3.0.2 \
    multiprocess==0.70.15 \
    nbclient==0.10.0 \
    nbconvert==7.16.4 \
    nbformat==5.10.4 \
    nest-asyncio==1.6.0 \
    networkx==3.3 \
    notebook_shim==0.2.4 \
    numpy==1.24.3 \
    oauthlib==3.2.2 \
    open_clip_torch==2.26.1 \
    openai==1.54.4 \
    opencv-python==4.10.0.84 \
    orjson==3.10.10 \
    overrides==7.7.0 \
    packaging==24.1 \
    pandas==1.5.3 \
    pandocfilters==1.5.1 \
    peft==0.13.2 \
    Pillow==10.4.0 \
    requests==2.32.3 \
    scikit-learn==1.3.2 \
    sentence-transformers==3.2.1 \
    transformers==4.46.3

# Install git-based dependencies for LLaVA
RUN pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git@79ef45a6d8b89b92d7a8525f077c3a3a9894a87d

# Optional dependencies for training or building
RUN pip install git+https://github.com/bfshi/scaling_on_scales.git
RUN pip install deepspeed==0.12.6 ninja wandb flash-attn==2.6.2 --no-build-isolation
