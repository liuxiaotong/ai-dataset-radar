## Why
- Explain the user or business reason for this PR.

## What changed
- Summarize the concrete code / workflow changes.

## Risk
- Low / Medium / High:
- Rollback plan:

## Checks
- [ ] `uv run ruff check src tests mcp_server agent`
- [ ] `uv run pytest -q --deselect tests/test_llm_client.py::TestOpenAICompatibleProvider::test_basic_call`
- [ ] If `README.md` changed, confirm the website rebuild side effect is intended.
