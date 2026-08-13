# Process-witness experiment ledger

This directory records the execution and scientific decisions of the BonaFide process-witness
study.

- `experiment_log.jsonl` is the canonical append-only event ledger. Existing lines are never
  edited or reordered; corrections are new events that reference the superseded event ID.
- `EXPERIMENT_LOG.md` is the concise human-readable view. It may summarize several JSONL events,
  but it must not introduce a result or decision that has no canonical event.
- Large outputs remain in their frozen campaign directories. The ledger stores identities,
  hashes, job IDs, claim boundaries, and paths rather than copying artifacts into Git.

Every consequential decision, submission, failure, completion, audit, freeze, unblinding, and
claim change gets an event. Proposed decisions are recorded with `status="proposed"`; acceptance
or rejection is a later event. Retrospective entries are explicitly marked.

## JSONL event schema

Required fields:

```text
schema_version   fixed: adag.process-witness.experiment-event.v1
event_id         stable monotonic ID, pwexp-YYYYMMDD-NNN
recorded_at      RFC 3339 timestamp when the ledger entry was written
occurred_at      date or RFC 3339 time of the event itself
retrospective    whether this was recorded after the event
event_type       decision, execution, failure, audit, freeze, or observation
status           proposed, started, completed, passed, failed, superseded, or frozen
summary          one-sentence factual description
phase             experiment phase at the time
claim_boundary   what the event does and does not establish
```

Optional structured fields include `jobs`, `artifacts`, `code`, `inputs`, `metrics`,
`supersedes`, `notes`, and `next_actions`. Artifact entries should include a SHA-256 identity when
one exists. Never put credentials, environment secrets, raw model weights, or large responses in
the ledger.

