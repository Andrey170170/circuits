# Process-witness annotation guide

This guide describes the graph-blind token-painting workstation used before ADAG tracing. The
automatic colors are suggestions derived only from the frozen prompt and response text. They are
not claims about correctness, internal computation, causality, or faithfulness.

## Loading and saving

1. Open `scripts/bonafide/process_witness_annotation_review.html` in a current browser.
2. Select **Load data** and choose the single canonical `workstation-bundle.json` for the active
   annotation set. Do not load the `records/` directory for real review; those files are verbose
   diagnostic evidence.
3. Work on one response and one axis at a time. Changing the **General type** changes both the
   palette and the visible color layer.
4. Choose a value brush and click or drag across exact model tokens. A new manual color replaces
   the machine suggestion only on the active axis. Other axes remain independent and may overlap.
5. **Clear label** records an explicit unlabeled override. **Revert to machine** removes the manual
   override and restores the automatic suggestion. **Undo** reverses the latest local action.
6. Mark an axis reviewed only after inspecting the full response on that axis. Any later edit
   automatically reopens it.
7. Export regularly. The export is a provenance-bound, append-only review ledger; importing it
   with the same bundle resumes the session. An identity mismatch fails closed.

## The three levels that matter most

Most review decisions use three complementary axes:

| Level | Axis | Paint extent | Question it answers |
| --- | --- | --- | --- |
| Event | `process_span` | Whole clause, sentence, or compact line | What kind of candidate event is this text describing? |
| Event | `event_operation` | Same full event span | Which broad operation characterizes that event? |
| Token | `process_role` | Specific operands, outputs, states, or cues | What role does this token play inside the event? |

The exact-symbol `operation` axis is different from `event_operation`. For example, a sentence can
be `mixed_arithmetic` on `event_operation`, while its exact `*`, `+`, and `mod` tokens retain
separate colors on `operation`. Do not replace those local operator labels with the event-level
summary.

## Axis reference

### `discourse_phase`

This is the broad reading layer and the default view. It colors complete text units.

| Value | Choose it when the unit primarily… |
| --- | --- |
| `orientation_or_restating` | Restates the concrete input or situates the task without giving instructions. |
| `instruction_or_task_description` | Quotes or paraphrases what the task, rules, or user requires. |
| `planning` | Announces a future approach or ordering of work. |
| `reference_lookup_or_reading` | Reads a table, graph, mapping, list, or outgoing-reference inventory. |
| `working_or_derivation` | Carries out a local transformation, calculation, or state update. |
| `uncertainty_or_deliberation` | Weighs alternatives or expresses uncertainty. |
| `verification` | Checks a calculation, result, or completed chain. |
| `correction_or_reconsideration` | Identifies and revises a suspected mistake. |
| `conclusion` | States the answer or a settled result in prose. |
| `answer_serialization` | Is the exact terminal machine-readable answer object. This is structural and normally automatic. |
| `unclassified_or_other` | Does not fit a supported class, or evidence is too weak. Prefer this over guessing. |

### `process_span`

Use this for a full candidate event. A label ending in `_candidate` describes the text, not proof
that the model internally executed it.

| Value | Use for… |
| --- | --- |
| `arithmetic_event_candidate` | A locally expressed arithmetic calculation or result-producing formula. |
| `state_transition_event_candidate` | An executed move or update from one state/node to another. |
| `state_update_with_arithmetic` | A state update whose new value is computed arithmetically. |
| `state_relation_or_schema_candidate` | An arrow, edge, or relation presented as task data/schema rather than an executed move. |
| `reference_lookup_event_candidate` | Reading or selecting an entry from task data. |
| `comparison_or_selection_event_candidate` | Comparing options or selecting an extremum/winner. |
| `encoding_or_decoding_event_candidate` | An active encode, decode, reverse, swap, substitute, or shift transformation. |
| `verification_event_candidate` | A check of a result or completed work. |
| `correction_event_candidate` | A local correction or recomputation. |
| `answer_event_candidate` | A prose conclusion or exact terminal answer event. |

Do not use `state_transition_event_candidate` for a quoted rule such as “move to the referenced
section,” or for a bare edge such as `Digital Ethics → Quantum Physics`; use the description phase
and `state_relation_or_schema_candidate` instead.

### `event_operation`

This summarizes a whole process span. Values include arithmetic operations, `mixed_arithmetic`,
`state_transition`, `state_relation_or_schema`, `lookup`, `order_comparison`,
`encoding_or_decoding`, `verification`, and `correction`. Choose the main operation of the event;
when several arithmetic operations are inseparable within one unit, use `mixed_arithmetic`.

### `operation`

This is the exact local cue layer. Paint only the literal operator or operation word: `+`, `-`,
`*`, `/`, `mod`, `verify`, `lookup`, and similar cues. It is intentionally sparse. A whole sentence
should usually be painted on `event_operation`, not here.

### `process_role`

This is the main fine-grained token layer.

| Value | Use for… |
| --- | --- |
| `operator_cue` | The literal token that signals an operation. |
| `input_operand` / `operand_candidate` | A value consumed by the local event. Use `candidate` when syntax is suggestive but role is uncertain. |
| `intermediate_result` / `intermediate_result_candidate` | A produced non-final value, normally an explicit scalar RHS or yielded value. Do not label formula coefficients or initial state merely because they follow an earlier `=`. |
| `state_value` / `state_value_candidate` | The source/current state named in a transition or relation. |
| `state_update` | The destination/new state or updated running value. |
| `final_result` | The actual presented final value. The automatic draft assigns this conservatively from accepted terminal answer fields. |
| `answer_commitment` | A token in the exact accepted terminal answer field. |
| `verification_cue`, `correction_cue`, `planning_cue`, `uncertainty_cue` | A local discourse cue with that function. |
| `unknown` | A role is present but cannot be resolved reliably. |

Values without `_candidate` are available for explicit human confirmation; an automatic candidate
does not become confirmed merely because it looks plausible.

### `domain`

Choose the semantic domain best supported by the local text: `arithmetic`, `graph_or_state`,
`symbolic_text`, `comparison_or_game`, or `metacognitive`. Leave it unpainted when the unit mixes
domains or the choice would be arbitrary.

### `representation`

This records observable token form with some structural interpretation: operator/equality symbols,
answer keys and values, and numeric relation-result candidates. It is mostly machine-generated and
usually needs only error correction.

### `surface_form`

This records literal form such as numbers, whitespace, punctuation, Markdown, brackets, arrows,
and thinking boundaries. `compound_surface` means one tokenizer token covers more than one surface
class. This axis is a tokenizer/form reference, not a semantic judgment.

### `serialization_segment`

This observable layer separates `thinking_segment`, `final_answer_segment`, and
`boundary_or_control`. It should follow the exact stored assistant serialization. The two historical
dense reconstructions have reasoning-only scope and therefore do not receive a fabricated final
answer segment.

### `event_token_position`

This derived layer marks `span_onset`, `span_interior`, `span_terminal`, and a possible
`following_separator` relative to the current automatic process span. Because its boundaries depend
on a hypothesized event, treat it as a machine suggestion rather than raw observable truth.

### `usage`

This is human-only and controls later target selection:

| Value | Downstream meaning |
| --- | --- |
| `process_atlas_fit` | Eligible for the balanced process-atlas target bank. |
| `surface_reference` | Explicit form/punctuation comparison target; excluded from process-atlas fitting and labeling. |
| `trajectory_only` | Retained for ordered dense projection but not used to fit the primary atlas. |
| `unknown` | Review is unresolved; not eligible for freeze. |

The unpainted complement is never automatically “nuisance.” It may contain delayed, implicit, or
unrecognized computation.

### `event_status`

This is human-only and independent of semantic type: `correct`, `incorrect`, `attempted`,
`uncertain`, `not_applicable`, or `unknown`. Judge the described textual event against the task;
this still does not establish whether the model internally executed it.

## Long responses and long colored spans

The frozen cohort contains full reasoning responses, not short excerpts. The current 188 responses
contain 842,007 response tokens: mean 4,479, median 3,892, interquartile range 2,967–5,883, and
range 960–10,580. Broad automatic event spans are normally much shorter than a response. In the
frozen v8 draft, median runs are 10 tokens for `process_span`, 10 for `event_operation`, 11 for
`discourse_phase`, and one token for both exact `operation` and fine-grained `process_role`.
Rare unpunctuated lines remain large outliers (up to 794 tokens for an event span), so long spans
must be split or target-sampled during review rather than traced wholesale.

A painted span is not a command to trace every token in that span. After review, a separate frozen
conversion step will select a small response-balanced set of targets—such as a result token, state
update, or event onset—from `process_atlas_fit` regions. Broad responses receive only those selected
targets plus a balanced surface-reference panel. The two short dense discovery responses are the
only all-token trajectories, and even those remain subject to the strict-T5 total-context resource
gate.

## Review order

For each response, a practical order is:

1. Read the prompt/task context.
2. Review `discourse_phase` to separate instructions, lookup, work, checking, correction, and
   conclusion.
3. Review `process_span`, then `event_operation`.
4. Review fine-grained `process_role` at candidate events and answers.
5. Correct exact `operation`, `representation`, or `surface_form` only where needed.
6. Assign `event_status` and `usage` last, after the event boundary and role are settled.
7. Mark each completed response/axis reviewed and export the ledger.

Do not use ADAG traces, neuron identities, clusters, generated labels, or an attractive prospective
witness while annotating.
