"""Modal container image definitions.

Separate images keep image build times fast by only including the
dependencies actually needed by each function.
"""

import modal

# Base image used by Ollama-backed model functions.
# Installs the Ollama server binary and the Python client library.
ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "ollama>=0.4.0",
    )
    .add_local_dir("src/vecinita", remote_path="/root/vecinita")
)
