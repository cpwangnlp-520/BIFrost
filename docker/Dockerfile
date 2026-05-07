FROM nvcr.io/nvidia/pytorch:26.03-py3

WORKDIR /workspace/BIFrost

COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["/bin/bash"]
