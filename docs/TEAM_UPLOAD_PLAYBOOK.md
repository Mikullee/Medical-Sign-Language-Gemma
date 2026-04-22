# Team Upload Playbook

This document defines exactly where to put files and what to push, so each member can continue work without blocking others.

## Branch Roles

1. `dev-agent-sync`
- Purpose: collect new raw sequence videos
- Allowed changes:
  - `data/videos/raw_sequences/**`
  - `data/annotations/sequence_recording_manifest_300.csv`
- Do not push approved data here

2. `main`
- Purpose: stable training-ready data
- Allowed changes:
  - `data/videos/approved_sequences/**`
  - `data/annotations/sequence_review_checklist_300.csv`
  - `data/annotations/sequence_annotations_300_template.csv`
- Merge by PR only

## Required Folder Targets

1. New recording (not reviewed yet)
- `data/videos/raw_sequences/<sequence_id>/`

2. Passed review
- `data/videos/approved_sequences/<sequence_id>/`

3. Need rerecord
- `data/videos/rerecord_needed/<sequence_id>/`

## Batch Workflow

1. Recorder
- Upload raw videos to `raw_sequences`
- Update `sequence_recording_manifest_300.csv` with status

2. Reviewer
- Mark result in `sequence_review_checklist_300.csv`
- Move passed files to `approved_sequences`
- Move failed files to `rerecord_needed`

3. Annotator
- Fill `start_sec`/`end_sec` in `sequence_annotations_300_template.csv`

4. Pusher
- Create one PR for one batch only
- PR must include evidence links and changed scope

## Naming Rules

1. Sequence video name format
- `SEQ##_X_##.mp4`
- Example: `SEQ01_A_03.mp4`

2. No spaces, no random suffixes
- Avoid names like `final2`, `new`, `copy`

## Commit Message Format

Use structured prefix:
- `data: add raw batch ...`
- `data: promote approved batch ...`
- `anno: update timestamp annotation for batch ...`
- `docs: update playbook/template ...`

## Minimal Push Checklist

1. Confirm branch is correct (`dev-agent-sync` or feature branch from `main`)
2. Confirm only expected folders are changed
3. Confirm no temp files are staged
4. Confirm commit message uses structured prefix
5. Push and open PR
