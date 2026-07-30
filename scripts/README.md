# Scripts

See [docs/typer-cli-guide.md](docs/typer-cli-guide.md) for Typer CLI conventions (`@app.callback()`, commands, invocation style).

Scripts share reusable helpers from the `src.utils` package:

| Helper | Module | Description |
|---|---|---|
| `read_jsonl` | `src.utils.io` | Load and validate records from a JSONL file (msgspec decode + Pydantic validate) |
| `write_jsonl` | `src.utils.io` | Write Pydantic records to a JSONL file, one per line |
| `strip_html` | `src.utils.text` | Strip HTML tags and decode entities to plain text |

## 1. fetch_eval_data.py

Fetch QA-style data for RAG evaluation from GitHub Discussions or Stack Overflow.

### Commands

**GitHub — answered & resolved discussions**

```bash
uv run -m scripts.fetch_eval_data github [--url URL] [--num N] [--output PATH] [--category SLUG]
```

| Option | Default | Description |
|---|---|---|
| `--url` | `https://github.com/fastapi/fastapi` | GitHub repo URL |
| `--num` | `30` | Number of discussions to fetch |
| `--output` | `data/fastapi_discussions.jsonl` | Output path |
| `--category` | `questions` | Discussion category slug |

**Stack Overflow — answered questions by tag**

```bash
uv run -m scripts.fetch_eval_data stackoverflow [--url URL] [--num N] [--output PATH] [--tag TAG]
```

| Option | Default | Description |
|---|---|---|
| `--url` | `https://stackoverflow.com` | Stack Exchange site URL |
| `--num` | `30` | Number of questions to fetch |
| `--output` | `data/fastapi_stackoverflow.jsonl` | Output path |
| `--tag` | `fastapi` | Tag to filter by |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `GITHUB_READ_ACCESS` | Yes (GitHub only) | GitHub personal access token |
| `STACK_EXCHANGE_READ_ACCESS` | No | Stack Exchange API key (higher rate limit) |

## 2. normalize_eval_data.py

Combine fetched GitHub and Stack Overflow datasets into a unified eval JSONL.

```bash
uv run -m scripts.normalize_eval_data normalize [--github PATH] [--stackoverflow PATH] [--output PATH]
```

| Option | Default | Description |
|---|---|---|
| `--github` | `data/fastapi_discussions.jsonl` | GitHub discussions JSONL input |
| `--stackoverflow` | `data/fastapi_stackoverflow.jsonl` | Stack Overflow questions JSONL input |
| `--output` | `data/eval_dataset.jsonl` | Unified output path |

Each record in the unified output has fields `id`, `source`, `title`, `url`, `body`, `answerText`, `createdAt`, `score`, `answerScore`, and `tags`.

## 3. classify_eval_data.py

Classify eval records from `data/eval_dataset.jsonl` into one of three retrieval-difficulty labels using an LLM via OpenRouter.

```bash
uv run -m scripts.classify_eval_data classify [--input-path PATH] [--output-path PATH]
```

| Option | Default | Description |
|---|---|---|
| `--input-path` | `data/eval_dataset.jsonl` | Unified eval records to classify |
| `--output-path` | `data/eval_dataset_labeled.jsonl` | Output path with predicted labels |

Each output record adds a `label` field with one of the following values:

| Label | Meaning |
|---|---|
| `DIRECT_LOOKUP` | Answerable from a single concrete fact in the docs |
| `MULTI_HOP` | Requires combining multiple features or concepts |
| `CONCEPTUAL` | About design, architecture, trade-offs, or best practices |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `OPENROUTER_BASE_URL` | Yes | OpenRouter base URL (e.g. `https://openrouter.ai/api/v1`) |
