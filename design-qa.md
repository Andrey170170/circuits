# Process-witness token painter design QA

- Source of truth: user Excalidraw sketch at
  `/scratch/local/u1653998/1795966/chpc-codex/capsules/t3/t3/userdata/attachments/b600b237-482e-4b95-bdb7-113bc30ba349-0f379120-6778-4a6a-9c17-053f606fa828.png`
- Implementation: `scripts/bonafide/process_witness_annotation_review.html`
- Canonical data: `process-witness-graph-blind-auto-v7/workstation-bundle.json`
- Bundle SHA-256: `aaaf11778aa0071cb9c754ce214d594215074c83fd8f9d200f38c0bbb4228ced`
- Browser evidence: `/tmp/pw-v7-default-discourse.png`, `/tmp/pw-v7-process-span.png`,
  `/tmp/pw-v7-event-operation.png`, `/tmp/pw-v7-operation.png`,
  `/tmp/pw-v7-after-drag-paint.png`, and `/tmp/pw-v7-resumed-review.png`
- Browser report: `/tmp/pw-v7-real-qa-report.json`
- Viewport: 1440 by 813, headless Google Chrome fallback
- State captured: real 188-response bundle loaded; broad and exact semantic axes inspected;
  scrolling exercised; four manual token overrides; one response/axis marked reviewed; exported
  review re-imported.

The product-native T3 preview reported that browser preview was unavailable in this environment,
so QA used the permitted headless-Chrome fallback. The real 39,424,274-byte bundle loaded and
validated all 188 responses in 4.984 seconds. An actual wheel event moved the inner document
scroller from 0 to 900 pixels, with 10,092 pixels of available scroll range. The run exercised
prompt/task display, axis switching, click-and-drag painting, clear, revert, undo, review
completion, export, and fresh-page resume with no page or console errors.

The implementation follows the sketch's interaction model rather than its literal drawing style:
one response occupies the center, the active general type controls the right-hand palette and the
visible token layer, and each authoritative model token carries at most one color within an axis.
Different axes can overlap. Machine paint is visible immediately; manual paint overrides it;
clear and revert-to-machine are separate operations.

No P0, P1, or P2 visual or interaction issue remains in the tested desktop state. The default
`discourse_phase` layer visibly colors broad response structure, while `process_span`,
`event_operation`, and exact `operation` show progressively narrower paint. The prompt is
collapsed by default to keep the response usable; its task identity remains visible in the header.
The right palette intentionally scrolls independently for axes with many values.

Final result: passed
