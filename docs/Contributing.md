# Contributing to SmolLM3 Fine-tuning Pipeline

Thanks for contributing! This project is an end-to-end fine-tuning pipeline for SmolLM3 with Trackio monitoring and Hugging Face integration.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements/requirements.txt
```

## Conventions
- PEP 8, type hints, descriptive names, f-strings
- Dataclass configs, validate in `__post_init__`, YAML and Python supported
- Try/except for external APIs (HF, Trackio) with user-friendly messages
- Trackio: include URL and experiment name; log metrics; save checkpoints
- Structured logging with consistent fields

## Project map
- `config/`: training configs (inherit from `SmolLM3Config`)
- `src/`: `train.py`, `model.py`, `data.py`, `trainer.py`, `monitoring.py`
- `scripts/`: deployment, datasets, model pushing, quantization
- `templates/`: Spaces and templates
- `tests/`: unit + integration
- `docs/`: docs

## Testing
```bash
pytest -q
```
Add focused unit tests and at least one integration test if you change pipeline behavior.

## Submitting changes
1. Branch: `feat/<short-name>` or `fix/<short-name>`
2. Conventional commits: `type(scope): message`
3. Open PR using template; link issues (`Fixes #<id>`)
4. Add labels (area, difficulty, priority, status)

## PR checklist
- Type hints + docstrings
- Tests updated and passing
- Docs updated in `docs/` (and README if user-facing)
- No secrets/tokens; use env vars
- Consider performance/memory (A100/H100 paths)

## Labels
- Areas: `dataloader`, `quantization`, `trainer`, `interface`, `config`, `monitoring`, `deployment`, `datasets`, `model`, `templates`, `tests`
- Difficulty: `good first issue`, `good second issue`, `easy`, `medium`, `expert`
- Priority: `priority: low|medium|high|critical`
- Status: `status: needs triage|needs info|blocked|in progress|help wanted`

## Data & security
- Validate/sanitize datasets; prefer HF Datasets
- Never hardcode tokens; validate via env vars

## Performance tips
- Mixed precision and gradient checkpointing
- Tune batch size and accumulation per hardware
- Use Trackio for metrics and artifacts

## Troubleshooting
- CUDA OOM: lower batch size / increase grad accumulation
- Network: retry and verify connectivity
- HF token: ensure read/write scope

## Release & push
- Use `scripts/model_tonic/push_to_huggingface.py`
- Include model card and training configuration
