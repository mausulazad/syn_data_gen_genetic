# FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
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

# Install the project dependencies without version specifications
RUN pip install \
    absl-py accelerate aiohappyeyeballs \
    aiohttp aiosignal annotated-types \
    anyio argon2-cffi argon2-cffi-bindings \
    arrow asttokens async-lru \
    async-timeout attrs av \
    babel backoff beautifulsoup4 \
    bitsandbytes bleach boto3 \
    botocore cachetools certifi \
    cffi charset-normalizer click \
    cloudpickle comm contourpy \
    cycler dask datasets \
    debugpy decorator defusedxml \
    dill distro einops \
    einops-exts evaluate exceptiongroup \
    executing fastapi fastjsonschema \
    filelock fire flash-attn \
    fonttools fqdn frozenlist \
    fsspec ftfy fuzzywuzzy \
    google-auth google-auth-oauthlib greenlet \
    grpcio h11 httpcore \
    httpx huggingface-hub idna \
    importlib_metadata ipykernel ipython \
    ipywidgets isoduration jedi \
    Jinja2 jiter jmespath \
    joblib json5 jsonpatch \
    jsonpointer jsonschema jsonschema-specifications \
    jupyter_client jupyter_core jupyter-events \
    jupyter-lsp jupyter_server jupyter_server_terminals \
    jupyterlab jupyterlab_pygments jupyterlab_server \
    jupyterlab_widgets keras kiwisolver \
    langchain langchain-core langchain-ollama \
    langchain-openai langchain-text-splitters langsmith \
    Levenshtein lightning lightning-cloud \
    lightning_sdk lightning-utilities litdata \
    litserve llava llm2vec \
    locket Markdown markdown-it-py \
    MarkupSafe matplotlib matplotlib-inline \
    mdurl mistune mpmath \
    multidict multiprocess nbclient \
    nbconvert nbformat nest-asyncio \
    networkx notebook_shim numpy \
    oauthlib ollama open_clip_torch \
    openai opencv-python orjson \
    overrides packaging pandas \
    pandocfilters parso partd \
    peft pexpect pillow \
    platformdirs prometheus_client prompt_toolkit \
    protobuf psutil ptyprocess \
    pure_eval py-cpuinfo pyarrow \
    pyarrow-hotfix pyasn1 pyasn1_modules \
    pycparser pydantic pydantic_core \
    Pygments PyJWT pyparsing \
    python-dateutil python-json-logger python-Levenshtein \
    python-multipart pytorch-lightning pytz \
    PyYAML pyzmq qwen-vl-utils \
    RapidFuzz referencing regex \
    requests requests-oauthlib requests-toolbelt \
    rfc3339-validator rfc3986-validator rich \
    rpds-py rsa s3transfer \
    safetensors scikit-learn scipy \
    seaborn Send2Trash sentence-transformers \
    sentencepiece setuptools shapely \
    simple-term-menu six sniffio \
    soupsieve SQLAlchemy stack-data \
    starlette sympy tenacity \
    tensorboard tensorboard-data-server termcolor \
    terminado threadpoolctl tiktoken \
    timm tinycss2 tokenizers \
    tomli toolz torch \
    torchmetrics torchvision tornado \
    tqdm traitlets transformers \
    triton types-python-dateutil typing_extensions \
    tzdata ultralytics ultralytics-thop \
    uri-template urllib3 uvicorn \
    wcwidth webcolors webencodings \
    websocket-client Werkzeug wheel \
    widgetsnbextension xxhash yarl \
    zipp

# Optional dependencies for training or building
RUN pip install git+https://github.com/bfshi/scaling_on_scales.git
RUN pip install deepspeed==0.12.6 ninja wandb
RUN pip install build twine
RUN pip install flash-attn==2.6.2 --no-build-isolation
