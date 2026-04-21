# Commit Convention

Use structured commit messages to keep history reviewable.

## Format

`<type>: <short summary>`

Examples:
- `data: add approved sequence videos for 09C-12C`
- `exp: tune mixed BiGRU lr=3e-4 wd=1e-4`
- `eval: add gate-a 10-sentence result table`
- `docs: update weekly goals and workflow rules`
- `fix: correct dev-agent-sync branch link in readme`

## Allowed Types

- `data` data collection/annotation changes
- `exp` model training/experiment changes
- `eval` evaluation/reporting changes
- `docs` documentation-only changes
- `fix` bug fixes or logic fixes
- `chore` maintenance/non-functional updates

## Scope Rule

- One commit should represent one atomic change.
- Avoid `Update ...` as commit message.
