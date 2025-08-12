### Summary
Describe the change and why it’s needed. Reference related design/docs if any.

### Related issues
Link issues using `Fixes #123`, `Closes #456` where applicable.

### Type of change
- [ ] Bug fix
- [ ] Feature / Enhancement
- [ ] Documentation
- [ ] Performance / Memory
- [ ] Refactor
- [ ] CI / Tooling
- [ ] Tests
- [ ] Chore

### Components impacted
- [ ] config
- [ ] data / dataloader
- [ ] datasets
- [ ] model
- [ ] trainer
- [ ] training orchestrator (`src/train.py`)
- [ ] monitoring / trackio
- [ ] deployment / scripts
- [ ] templates / spaces
- [ ] interface
- [ ] quantization
- [ ] tests

### Estimated difficulty
- [ ] good first issue
- [ ] good second issue
- [ ] easy
- [ ] medium
- [ ] expert

### How to test
Provide commands and steps. Include dataset details, config used, and expected results.

```bash
# examples (edit as needed)
python scripts/training/train.py --config config/train_smollm3_openhermes_fr.py
pytest -q
```

### Screenshots / Logs
If applicable, add logs or images that demonstrate outcomes.

### Checklist
- [ ] Follows Python style (PEP 8), adds type hints and docstrings
- [ ] Uses descriptive variable names and f-strings
- [ ] Error handling added for external API calls (HF, Trackio) with clear messages
- [ ] Structured logging used consistently
- [ ] Tests added/updated and pass locally (`pytest -q`)
- [ ] Documentation updated under `docs/` and README if user-facing
- [ ] If editing configs, includes Trackio URL and experiment name; logs metrics every N steps
- [ ] No secrets or tokens committed; uses environment variables
- [ ] If dataset code changed: validates and filters sensitive data
- [ ] If quantization changed: tested scripts under `scripts/model_tonic/`
- [ ] If trainer/orchestrator changed: verified memory/perf impact and checkpoint saving
- [ ] Labels added (area, difficulty, priority, status) as appropriate

### Breaking changes
Call out any breaking changes and migration steps.

### Additional context
Anything else that reviewers should know.


