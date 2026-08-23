# Trace Observatory v1 design system

Source concept: `trace-observatory-v1-concept.png` (1600 by 1000). The concept fixes the
information hierarchy and visual language. Runtime trace values always replace illustrative
numbers in the concept.

## Layout

- 56 px top command bar.
- 240 px target rail on desktop; collapsible below 1100 px.
- Fluid center canvas with a 280-320 px evidence inspector.
- A target-local token window and filters sit above the canvas; the full prefix is explicitly
  expandable, and provenance is a bottom drawer.
- Use open rails, dividers, tables, and canvas. Do not wrap every region in a card.

## Color

- Page and canvas: true white, `#ffffff`.
- Secondary rail and layer bands: `#f7f9fc` and `#f1f5f9`.
- Primary text: `#111827`; secondary text: `#526075`; quiet text: `#7b8798`.
- Borders/grid: `#d9e0e8`; strong dividers: `#c4ced9`.
- Primary/positive: `#175cff`; hover/select wash: `#edf3ff`.
- Negative: `#f04f5f`; warning: `#ea580c`.
- No gradients, glow, glass, or warm off-white substitution.

## Type

- UI chrome and prose: Inter-like system sans, `Inter, ui-sans-serif, system-ui`.
- Identities, positions, values, and tokens: `IBM Plex Mono, ui-monospace, monospace`.
- App title 18/24 semibold; section title 12/16 semibold; controls 12/16 medium; body 13/20;
  compact data 11/16.

## Geometry

- One-pixel borders; 4 px control radius and 6 px node/panel radius.
- Spacing scale: 4, 8, 12, 16, 24, 32 px.
- Selected rows use a 3 px cobalt left rule and pale-blue fill.
- Shadows only for temporary overlays; persistent panes use borders.

## Components and states

- Target row: position, token, local context, graph size; default/hover/selected/error states.
- Token strip: normal text flow, token boundary on hover, prediction boundary, selected target.
- Filter toolbar: layer, token position, input-token visibility, polarity, magnitude, retained
  mass, edge budget, and neuron search.
- Graph canvas: layer bands, compact raw-neuron nodes, signed red/blue edges, zoom and reset.
- Evidence inspector: raw identity first; label methods appear in parallel columns below it.
- Provenance drawer: artifact, raw/display counts, retained mass, hashes, and claim warning.
- Minimal 1.5 px outline icons for menu, filter, search, zoom, reset, save, and disclosure.

## Allowed first-viewport copy

- `Trace Observatory`
- Actual model ID and selected `position ... · token ...`
- `Raw identity`, installed label-set names, and `Save workspace`
- `Target trajectory`, `Reasoning tokens`, `Prediction boundary`, `Neuron evidence`
- Filter names from the implementation plan
- Actual field names, values, connection headings, and provenance fields
- `Pruned local attribution graph — not a computation transcript.`

No marketing copy, login UI, graph generation, steering, cloud controls, fake metrics, badges, or
decorative product areas are allowed in v1.

## Responsive behavior

- At 1100-1300 px, narrow both rails and retain the graph as the largest region.
- Below 1100 px, target and evidence rails become drawers; context, filters, and canvas remain.
- Below 760 px, filters wrap into two rows and evidence opens full-width over the canvas.
- Keyboard focus, selected state, and sign information must never rely on color alone.
