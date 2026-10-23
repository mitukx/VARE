FROM python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
