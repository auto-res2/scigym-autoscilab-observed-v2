# Base image pinned to a version AND its digest.
FROM python:3.11.16-slim-trixie@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libglib2.0-0 と X 系は pygraphviz の wheel が同梱しない共有ライブラリ
RUN apt-get update && apt-get install -y \
    git curl make build-essential libglib2.0-0 libx11-6 libxext6 libxrender1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# 公式 LLM-AutoSciLab は pyproject を持たず pip / uv で入らないので、record の repositories と同じコミットを clone して PYTHONPATH に乗せる
# （LICENSE が無いのでリポジトリにはコピーしない）。実行時の hook が import されたファイルのハッシュを snapshot と突き合わせる
ARG AUTOSCILAB_COMMIT=351aae73e6c6c279e67943b77143b0f118d1e289
RUN git clone --quiet https://github.com/scientific-discovery/LLM-AutoSciLab /opt/autoscilab \
    && git -C /opt/autoscilab checkout --quiet "$AUTOSCILAB_COMMIT" \
    && rm -rf /opt/autoscilab/.git
ENV PYTHONPATH=/opt/autoscilab

# uv pinned to a version AND its digest.
COPY --from=ghcr.io/astral-sh/uv:0.12.6@sha256:88bc6eb1ccd4b82efd0e1b530caffabddf50dc2bf612e66c14ea25b8ee8a4d3d /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-cache --group eval
# tellurium.teconverters.convert_omex が import する libcombine の代用（OMEX は使わない。pyproject の exclude を参照）
RUN echo "# stub for tellurium import; OMEX archives are never used here" > .venv/lib/python3.11/site-packages/libcombine.py
ENV UV_NO_SYNC=1

COPY . .
RUN mkdir -p .research/results
CMD ["bash"]
