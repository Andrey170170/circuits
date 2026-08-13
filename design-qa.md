# Process-witness token painter design QA

- Source of truth: user Excalidraw sketch at
  `/scratch/local/u1653998/1795966/chpc-codex/capsules/t3/t3/userdata/attachments/b600b237-482e-4b95-bdb7-113bc30ba349-0f379120-6778-4a6a-9c17-053f606fa828.png`
- Implementation: `scripts/bonafide/process_witness_annotation_review.html`
- Canonical data: `process-witness-graph-blind-auto-v5/workstation-bundle.json`
- Browser evidence: `/tmp/process-witness-v5-real-qa.png`
- Browser report: `/tmp/process-witness-v5-real-qa-report.json`
- Viewport: 1440 by 768, headless Google Chrome fallback
- State captured: real 188-response bundle loaded; operation axis visible; three manual token
  overrides; one response/axis marked reviewed; exported review re-imported.

The product-native T3 preview reported that browser preview was unavailable in this environment,
so QA used the permitted headless-Chrome fallback. The real 32,862,635-byte bundle loaded and
validated all 188 responses in 3.213 seconds. The run exercised prompt/task display, axis
switching, click-and-drag painting, clear, revert, undo, review completion, export, and fresh-page
resume. All 16 assertions passed with no page or console errors.

The implementation follows the sketch's interaction model rather than its literal drawing style:
one response occupies the center, the active general type controls the right-hand palette and the
visible token layer, and each authoritative model token carries at most one color within an axis.
Different axes can overlap. Machine paint is visible immediately; manual paint overrides it;
clear and revert-to-machine are separate operations.

No P0, P1, or P2 visual or interaction issue remains in the tested desktop state. The prompt is
collapsed by default to keep the response usable; its task identity remains visible in the header.
The right palette intentionally scrolls independently for axes with many values.

Final result: passed
