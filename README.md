# vecinita-model

Serverless LLM model hosting on [Modal](https://modal.com).  
Supports Ollama-compatible models (Llama 3, Mistral, Phi-3, Gemma 2, and more) with a simple REST API.

---

## Project layout

```
src/vecinita/
├── app.py          # Modal application entry-point (deploy / serve this file)
├── config.py       # Settings & supported model registry
├── images.py       # Container image definitions
├── volumes.py      # Persistent Modal volume for model weights
├── models/
│   ├── base.py     # Abstract model backend interface
│   └── ollama.py   # Ollama model backend implementation
└── api/
    ├── routes.py   # FastAPI route handlers  (/health, /chat, /stream)
    └── schemas.py  # Pydantic request / response schemas
tests/
├── conftest.py
├── test_schemas.py
└── test_routes.py
```

---

## Prerequisites

- Python ≥ 3.11
- A [Modal](https://modal.com) account (`pip install modal && modal setup`)
- (Optional) A local [Ollama](https://ollama.com) install for local testing

---

## Local development setup

```bash
# 1. Clone the repo and create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install the package and dev dependencies
pip install -e ".[dev]"

# 3. Copy the example env file
cp .env.example .env

# 4. Run tests
pytest

# 5. Serve locally (hot-reload, no GPU required)
modal serve src/vecinita/app.py
```

---

## Preloading model weights

Model weights are stored in a Modal persistent volume (`vecinita-models`).  
Run the following **once per model** to download weights into that volume:

```bash
# Download Llama 3.2 (default)
modal run src/vecinita/app.py::download_model --model-name llama3.2

# Download Mistral 7B
modal run src/vecinita/app.py::download_model --model-name mistral

# Download Phi-3
modal run src/vecinita/app.py::download_model --model-name phi3
```

Supported model IDs are defined in `src/vecinita/config.py`:

| Model ID       | Description                  |
|----------------|------------------------------|
| `llama3.2`     | Meta Llama 3.2 (default)     |
| `llama3.2:1b`  | Meta Llama 3.2 1B (small)    |
| `llama3.1`     | Meta Llama 3.1               |
| `llama3.1:8b`  | Meta Llama 3.1 8B            |
| `mistral`      | Mistral 7B                   |
| `phi3`         | Microsoft Phi-3              |
| `gemma2`       | Google Gemma 2               |
| `gemma2:2b`    | Google Gemma 2 2B (small)    |

---

## Deploying to Modal

```bash
modal deploy src/vecinita/app.py
```

After deployment Modal prints a URL like  
`https://<your-workspace>--vecinita-model-api.modal.run`.

---

## API reference

### `GET /health`

Returns service status and the list of models cached in the volume.

```json
{
  "status": "ok",
  "models": ["llama3.2", "mistral"]
}
```

---

### `POST /chat`

Send a conversation and receive a **complete** response.

**Request body**

```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "What is 2 + 2?"}
  ],
  "temperature": 0.7,
  "max_tokens": null
}
```

**Response**

```json
{
  "model": "llama3.2",
  "message": {"role": "assistant", "content": "4"},
  "done": true
}
```

---

### `POST /stream`

Stream response tokens as [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).

**Request body** — same as `/chat`.

**Event stream** (each line is one SSE event):

```
data: {"model":"llama3.2","content":"The","done":false}
data: {"model":"llama3.2","content":" answer","done":false}
data: {"model":"llama3.2","content":" is 4.","done":true}
```

---

## Adding a new model

1. Add an entry to `SUPPORTED_MODELS` in `src/vecinita/config.py`.
2. Run `modal run src/vecinita/app.py::download_model --model-name <id>` to pull weights.
3. Redeploy with `modal deploy src/vecinita/app.py`.

---

## GPU configuration

By default the API function runs on CPU (suitable for small models and
development).  To enable GPU inference uncomment and adjust the `gpu=` line
in `src/vecinita/app.py`:

```python
@app.function(
    ...
    gpu=modal.gpu.A10G(),   # or T4, A100, H100, …
)
```

---

## Running tests

```bash
pytest -v
```

Tests mock the Ollama client so no running server or GPU is needed.