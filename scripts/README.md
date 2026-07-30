# Scripts

Both scripts share reusable helpers from the `src.utils` package:

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
uv run python scripts/fetch_eval_data.py github [--url URL] [--num N] [--output PATH] [--category SLUG]
```

| Option | Default | Description |
|---|---|---|
| `--url` | `https://github.com/fastapi/fastapi` | GitHub repo URL |
| `--num` | `30` | Number of discussions to fetch |
| `--output` | `data/fastapi_discussions.jsonl` | Output path |
| `--category` | `questions` | Discussion category slug |

**Stack Overflow — answered questions by tag**

```bash
uv run python scripts/fetch_eval_data.py stackoverflow [--url URL] [--num N] [--output PATH] [--tag TAG]
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
uv run python scripts/normalize_eval_data.py normalize [--github PATH] [--stackoverflow PATH] [--output PATH]
```

| Option | Default | Description |
|---|---|---|
| `--github` | `data/fastapi_discussions.jsonl` | GitHub discussions JSONL input |
| `--stackoverflow` | `data/fastapi_stackoverflow.jsonl` | Stack Overflow questions JSONL input |
| `--output` | `data/eval_dataset.jsonl` | Unified output path |

Each record in the unified output has fields `id`, `source`, `title`, `url`, `body`, `answerText`, `createdAt`, `score`, `answerScore`, and `tags`.
