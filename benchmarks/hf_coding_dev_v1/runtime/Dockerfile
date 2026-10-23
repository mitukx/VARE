FROM vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b
USER root
COPY requirements.lock.txt /tmp/task-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/task-requirements.txt && python -m pip check
ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
USER 65534:65534
