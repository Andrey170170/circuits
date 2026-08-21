# Raw-graph observatory candidate review v2

Status: **accepted review extension**. This version extends the v1 candidate browser; it does not
authorize a trace launch or modify the main process-witness target-selection algorithm.

## Decision

Keep `outright-task-review-v1` immutable and publish a separate v2 review packet with:

- every labeled outright completion (`src_type` exactly `complex` or `graph`), including Qwen;
- exact response-token positions reconstructed with a pinned tokenizer and chat-serialization
  profile for each model;
- response and causal-prefix length statistics;
- one draft target and any number of saved targets per completion;
- optional comments and a source-bound JSON export suitable for target audit and discussion.

The completion-level candidate checkbox and token-level trace targets remain distinct. A saved
target identifies one observed response token at one zero-based response position. The UI reports
both the number of response tokens before it and the total teacher-forced causal-prefix length.

## Qwen boundary

V1 physically excluded Qwen to preserve an unopened review set. V2 intentionally includes Qwen
because this browser is exploratory tooling and the main campaign will use a separately designed,
frozen algorithm over a larger corpus rather than these human choices.

- Any Qwen completion viewed, selected, commented on, traced, or discussed through v2 is an opened
  exploratory case and must not later be described as held out.
- V2 choices and graph outcomes do not tune or validate the main automated target-selection
  protocol. That protocol must be named, frozen, and evaluated under its own firewall.
- Qwen Instruct and Thinking token identities remain separate, as do all model revisions.

## Token reconstruction contract

`BonaFide.csv` preserves response text and annotations but does not preserve authoritative original
generation token IDs and revisions for every model. V2 therefore makes a declared reconstruction:

1. pin a tokenizer snapshot, chat-template hash, system prompt, and serialization mode per model;
2. tokenize through the same teacher-forced helper used by tracing;
3. require the tokenized stored response to align exactly to source character offsets;
4. record response IDs, prefix IDs, profile hashes, and reconstruction status in the packet;
5. fail closed rather than substituting a tokenizer from a related model family.

For Thinking models the stored reasoning body is replayed after the historical generation prefix.
For Instruct models it is reconstructed as assistant content under the historical CoT system
prompt. In both cases, parsing of the released corpus may have stripped boundary whitespace,
thinking delimiters, the answer block, or other original bytes. These are reproducible positions
under the v2 profile, not recovered historical generation positions.

## Color and state semantics

- green: overlap with a localized faithful source annotation;
- red: overlap with a localized unfaithful source annotation;
- split red/green: overlap with both;
- yellow: current unsaved draft target;
- blue: saved trace target.

Draft and saved colors take visual precedence, while a separate polarity mark preserves annotation
evidence. Whole-CoT or unlocalized labels do not color every token. Color is always paired with a
text label, tooltip, or icon.

Only saved targets are exported. Each export preserves exact immutable target identity, token and
prefix positions, bounded local context, annotations, comment, completion identity, source hash,
review-payload hash, and tokenizer-profile hash. Draft state may persist locally but must be called
out as omitted when an export is requested.

## Gate before tracing

An exported v2 selection is a discussion artifact, not an execution manifest. Before tracing, a
consumer must validate the source and payload hashes, join every target ID back to the packet,
re-tokenize under the pinned profile, reject altered or duplicate identities, and freeze a new
trace manifest with code revision and trace configuration.
