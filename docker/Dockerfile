FROM nvcr.io/nvidia/pytorch:26.03-py3

WORKDIR /workspace/bif

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["/bin/bash"]
