import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { buildLayeredLayout } from "../circuits/observatory/assets/graph-layout.js";
import { profileDisplay, profileRows } from "../circuits/observatory/assets/profile-data.js";

const output = { id: "out", kind: "target_logit", token_text: "45" };
const mlp = (id, layer) => ({ id, kind: "raw_mlp_neuron", layer, neuron: id });
const input = (id, position) => ({ id, kind: "input_token", layer: -1, position });

test("model layers are strict vertical bands independent of graph topology", () => {
  const nodes = [mlp("l1", 1), output, mlp("l13", 13), mlp("l8", 8), input("in0", 0)];
  const edges = [
    { id: "skip", source: "l1", target: "out", attribution: 1 },
    { id: "back", source: "l13", target: "l8", attribution: 1 },
    { id: "input", source: "in0", target: "l13", attribution: 1 },
  ];

  const layout = buildLayeredLayout(nodes, edges);

  assert.deepEqual(layout.bands.map((band) => band.label), ["out", "L13", "L8", "L1", "input tokens"]);
  assert.ok(layout.bands.every((band, index) => index === 0 || band.top > layout.bands[index - 1].top));
  assert.ok(layout.nodeById.get("out").y < layout.nodeById.get("l13").y);
  assert.ok(layout.nodeById.get("l13").y < layout.nodeById.get("l8").y);
  assert.ok(layout.nodeById.get("l8").y < layout.nodeById.get("l1").y);
  assert.ok(layout.nodeById.get("l1").y < layout.nodeById.get("in0").y);
});

test("input tokens wrap in a bounded bottom ribbon without changing graph width", () => {
  const internal = [output, mlp("l2-a", 2), mlp("l2-b", 2), mlp("l0", 0)];
  const fewInputs = Array.from({ length: 3 }, (_, index) => input(`few-${index}`, index));
  const manyInputs = Array.from({ length: 205 }, (_, index) => input(`many-${index}`, index));
  const moreInputs = Array.from({ length: 305 }, (_, index) => input(`more-${index}`, index));
  const few = buildLayeredLayout([...internal, ...fewInputs], []);
  const many = buildLayeredLayout([...internal, ...manyInputs], []);
  const more = buildLayeredLayout([...internal, ...moreInputs], []);

  assert.equal(many.bounds.width, few.bounds.width);
  assert.equal(many.fitBounds.width, few.fitBounds.width);
  assert.ok(many.bounds.height > few.bounds.height);
  assert.ok(more.bounds.height > many.bounds.height);
  assert.ok(many.bounds.height > many.fitBounds.height);
  assert.equal(more.fitBounds.height, many.fitBounds.height);
  assert.equal(many.nodeRows.filter((node) => node.role === "input").length, 205);
  assert.equal(many.bands.at(-1).role, "input");

  const inputs = many.nodeRows.filter((node) => node.role === "input");
  for (let leftIndex = 0; leftIndex < inputs.length; leftIndex += 1) {
    const left = inputs[leftIndex];
    for (let rightIndex = leftIndex + 1; rightIndex < inputs.length; rightIndex += 1) {
      const right = inputs[rightIndex];
      const separated =
        left.x + left.width / 2 <= right.x - right.width / 2 ||
        right.x + right.width / 2 <= left.x - left.width / 2 ||
        left.y + left.height / 2 <= right.y - right.height / 2 ||
        right.y + right.height / 2 <= left.y - left.height / 2;
      assert.ok(separated, `${left.id} overlaps ${right.id}`);
    }
  }
});

test("input ribbon preserves ascending token position despite connectivity", () => {
  const inputs = [input("in2", 2), input("in0", 0), input("in4", 4), input("in1", 1), input("in3", 3)];
  const internal = Array.from({ length: 4 }, (_, index) => mlp(`mlp-${index}`, 0));
  const nodes = [output, ...internal, ...inputs];
  const edges = Array.from({ length: 5 }, (_, position) => ({
    id: `edge-${position}`,
    source: `in${position}`,
    target: internal[(position * 3 + 1) % internal.length].id,
    attribution: 1,
  }));

  const layout = buildLayeredLayout(nodes, edges);

  assert.deepEqual(
    layout.nodeRows.filter((node) => node.role === "input").map((node) => node.position),
    [0, 1, 2, 3, 4],
  );
});

test("simple routing retains every displayed edge including input to MLP edges", () => {
  const nodes = [output, mlp("l0", 0), input("in0", 0), input("in1", 1)];
  const edges = [
    { id: "to-out", source: "l0", target: "out", attribution: 1 },
    { id: "from-in0", source: "in0", target: "l0", attribution: 0.5 },
    { id: "from-in1", source: "in1", target: "l0", attribution: -0.25 },
  ];

  const layout = buildLayeredLayout(nodes, edges);

  assert.deepEqual(layout.edgeRows.map((edge) => edge.id), ["to-out", "from-in0", "from-in1"]);
  assert.ok(layout.edgeRows.every((edge) => /^M[-.\d]+,[-.\d]+ C/.test(edge.path)));
});

test("profile preparation retains every finite source value", () => {
  const source = Array.from({ length: 37 }, (_, index) => (index % 2 ? -index / 10 : index / 10));
  const rows = profileRows(source);

  assert.equal(rows.length, source.length);
  assert.deepEqual(rows.map((row) => row.key).sort((a, b) => a - b), source.map((_, index) => index));
  assert.equal(rows[0].key, 36);
});

test("profile display defaults to a bounded view and can reveal every value", () => {
  const source = Array.from({ length: 37 }, (_, index) => index);

  const collapsed = profileDisplay(source);
  const expanded = profileDisplay(source, { expanded: true });

  assert.equal(collapsed.rows.length, 18);
  assert.equal(collapsed.total, 37);
  assert.equal(collapsed.hiddenCount, 19);
  assert.equal(expanded.rows.length, 37);
  assert.equal(expanded.hiddenCount, 0);
});

test("input visibility choices explain nodes and profiles accessibly", async () => {
  const html = await readFile(new URL("../circuits/observatory/assets/index.html", import.meta.url), "utf8");

  assert.match(html, /<option value="hide" selected>Hidden<\/option>/);
  assert.match(html, /<option value="show">Shown<\/option>/);
  assert.match(html, /aria-describedby="input-token-help"/);
  assert.match(html, /id="input-token-help"[^>]*>Hiding input nodes affects only the graph display\./);
});
