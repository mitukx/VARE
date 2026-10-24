FROM vare-mlx-cpu@sha256:a377e54dc56cbbf2b40627966ce8ff7c5152a0d580175921f45cbdbeff30bfa7
COPY candidates/baseline.py /tmp/vare-sanitize/sample_utils.py
COPY sanitize_sample_utils.py /tmp/vare-sanitize/sanitize_sample_utils.py
RUN ["python", "/tmp/vare-sanitize/sanitize_sample_utils.py"]
