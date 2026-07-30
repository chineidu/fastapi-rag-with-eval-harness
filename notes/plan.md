# Plan

## Phase 0: Corpus Setup

- Clone the FastAPI repository locally for easy access to the English version of the documentation.
- Strip away any raw html and other non-markdown content that can confuse a naive chunker (if there are any).

## Phase 1: Eval Set

### Source Real Questions

- From GitHub issues using GitHub API.
- From Stack Overflow using Stack Overflow API (tag=fastapi).

### Categorize The Questions

- Direct Lookup (~15 questions): Questions where the answer is a direct quote from a single page of the documentation.
- Multi-hop (~15 questions): Questions that require combining information from multiple pages of the documentation.
- Conceptual/How-to (~10 questions): Questions that require synthesizing information from multiple pages to provide a comprehensive answer. This category is designed to test the system's ability to understand and explain complex concepts.
