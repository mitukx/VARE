# Sandbox image for the A3 calibration of the v2 learning-pipeline grader (synthetic fixture).
# Dependencies only: the repository under test is never installed; it is bind-mounted per call.
# Real tasks need one image per repository with that repository's locked dependencies in /opt/venv.
FROM python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir pytest==8.4.2 \
    && rm -rf /root/.cache
