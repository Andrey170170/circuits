# BonaFide external API authorization

Status: active project-wide authorization, recorded 2026-08-17.

The user explicitly authorizes this project to send the public BonaFide-derived task prompts and
model responses used by the process-witness experiments to the OpenAI API. The authorized material
is expected to contain no private data. This authorization covers full-context requests, including
complete task prompts and complete generated responses, when required by a frozen experiment
protocol.

This is a data-destination authorization, not an unrestricted execution or spending authorization.
Each API campaign must still:

- bind the exact public corpus inputs, provider, model, request bodies, source revision, and output
  schema before submission;
- preserve a run-specific cost estimate and hard authorization ceiling;
- keep API credentials secret and out of artifacts and logs;
- retain provider receipts, usage, failures, and immutable output provenance; and
- obtain separate authorization before sending private, embargoed, personally identifying, or
  non-BonaFide-derived data, or before using a different external provider.

The authorization does not change the scientific claim boundary: API-generated labels remain
model proposals and do not become ground truth, computation evidence, ADAG adequacy evidence, or
motif/witness evidence without the separately frozen evaluation procedure.
