# Repository audit

## Scope

Audit run on the current `Neuro_ML` repository to prepare it for a public, employer-facing profile.

## Key findings

### High priority

- Branding and naming are inconsistent across root and submodule documentation (`P4P` and `Neuro_ML` mixed).
- Root documentation was overly long, included outdated details, and was not optimised for portfolio reviewers.
- `config.yaml` contains machine-specific absolute paths and references to a previous folder name.
- `frontend/README.md` links to a previous owner repository.
- Existing test scripts include hard-coded absolute local paths from another machine and are not CI-safe.

### Medium priority

- The repository has no root CI workflow for automated lint/test checks.
- There is no single lightweight development dependency file for quality tooling.
- `CONTRIBUTING.md` references collaboration details tied to a prior team workflow.

### Low priority

- Some docs use informal style markers and visual symbols not suited to a professional public profile.
- Legacy branch naming/history appears broad and may need tidy-up in the remote repository UI.

## Actions completed in this pass

- Replaced root `README.md` with a concise public-facing version.
- Added a CI workflow for lint and test checks.
- Added minimal quality tooling scaffold (`pyproject.toml`, `requirements-dev.txt`).
- Added deterministic repository tests under `tests/`.

## Recommended next actions

- Align all subdirectory READMEs and references to current project ownership.
- Replace machine-specific absolute paths in tracked config files with template placeholders.
- Add `LICENSE` and modernise `CONTRIBUTING.md` for solo/public contribution expectations.
- Add a results section with representative figures and benchmark table once curation is complete.
