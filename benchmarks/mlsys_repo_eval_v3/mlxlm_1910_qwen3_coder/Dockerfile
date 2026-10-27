FROM vare-mlx-cpu@sha256:a377e54dc56cbbf2b40627966ce8ff7c5152a0d580175921f45cbdbeff30bfa7
COPY grader.lock.txt /tmp/grader.lock.txt
RUN python -m pip install --no-cache-dir --require-hashes --no-compile -r /tmp/grader.lock.txt && rm /tmp/grader.lock.txt
