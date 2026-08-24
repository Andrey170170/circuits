import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { buildLayeredLayout } from "../circuits/observatory/assets/graph-layout.js";
import { profileDisplay, profileRows } from "../circuits/observatory/assets/profile-data.js";

const output = { id: "out", kind: "target_logit", token_text: "45" };
const mlp = (id, layer, position, neuron = id) => ({ id, kind: "raw_mlp_neuron", layer, position, neuron });
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

test("nodes align to ascending token-position columns across layers", () => {
  const nodes = [
    { ...output, position: 9 },
    mlp("l3-p9", 3, 9, 30),
    mlp("l1-p2", 1, 2, 10),
    mlp("l1-p9", 1, 9, 20),
    input("in2", 2),
    input("in9", 9),
  ];
  const tokenTextByPosition = new Map([[2, " the"], [9, "answer"]]);

  const layout = buildLayeredLayout(nodes, [], { tokenTextByPosition });

  assert.deepEqual(layout.columns.map((column) => column.position), [2, 9]);
  assert.ok(layout.columns[0].x < layout.columns[1].x);
  assert.equal(layout.nodeById.get("l1-p2").x, layout.nodeById.get("in2").x);
  assert.equal(layout.nodeById.get("out").x, layout.nodeById.get("l3-p9").x);
  assert.equal(layout.nodeById.get("l3-p9").x, layout.nodeById.get("l1-p9").x);
  assert.equal(layout.nodeById.get("l1-p9").x, layout.nodeById.get("in9").x);
  assert.equal(layout.columns[0].displayTokenText, "␠the");
  assert.equal(layout.columns[1].tokenText, "answer");
});

test("token text contributes to column width so headers cannot overlap", () => {
  const tokenText = "abcdefghijklmnopqr";
  const layout = buildLayeredLayout([input("in4", 4)], [], {
    tokenTextByPosition: new Map([[4, tokenText]]),
  });

  assert.equal(layout.columns[0].displayTokenText, tokenText);
  assert.ok(layout.columns[0].width >= tokenText.length * 7 + 16);
});

test("repeated nodes pack deterministically inside one layer-position cell without overlap", () => {
  const repeated = [8, 3, 12, 1, 5, 11, 7, 4, 10, 6, 9, 2]
    .map((neuron) => mlp(`n-${neuron}`, 13, 7, neuron));
  const layout = buildLayeredLayout([{ ...output, position: 7 }, ...repeated], []);
  const rows = layout.nodeRows.filter((node) => node.bandKey === "layer:13");

  assert.deepEqual(rows.map((node) => node.neuron), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  assert.ok(rows.every((node) => node.columnKey === "position:7"));
  assert.ok(layout.columns[0].width > 12 * 72);
  for (let index = 1; index < rows.length; index += 1) {
    assert.ok(
      rows[index - 1].x + rows[index - 1].width / 2 <= rows[index].x - rows[index].width / 2,
      `${rows[index - 1].id} overlaps ${rows[index].id}`,
    );
  }
});

test("token-column layout has no node overlaps", () => {
  const nodes = [
    { ...output, position: 4 },
    mlp("a", 3, 1, 1),
    mlp("b", 3, 1, 2),
    mlp("c", 3, 4, 3),
    mlp("d", 1, 1, 4),
    mlp("e", 1, 4, 5),
    input("in1", 1),
    input("in4", 4),
  ];
  const rows = buildLayeredLayout(nodes, []).nodeRows;

  for (let leftIndex = 0; leftIndex < rows.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < rows.length; rightIndex += 1) {
      const left = rows[leftIndex];
      const right = rows[rightIndex];
      const separated =
        left.x + left.width / 2 <= right.x - right.width / 2 ||
        right.x + right.width / 2 <= left.x - left.width / 2 ||
        left.y + left.height / 2 <= right.y - right.height / 2 ||
        right.y + right.height / 2 <= left.y - left.height / 2;
      assert.ok(separated, `${left.id} overlaps ${right.id}`);
    }
  }
});

test("nodes without a usable token position use a deterministic final fallback column", () => {
  const nodes = [
    mlp("known", 2, 3, 9),
    mlp("missing-b", 2, undefined, 8),
    mlp("missing-a", 2, "not-a-position", 4),
  ];
  const layout = buildLayeredLayout(nodes, []);

  assert.deepEqual(layout.columns.map((column) => column.position), [3, null]);
  assert.equal(layout.columns.at(-1).key, "unknown");
  assert.deepEqual(
    layout.nodeRows.filter((node) => node.columnKey === "unknown").map((node) => node.id),
    ["missing-a", "missing-b"],
  );
  assert.ok(Number.isFinite(layout.nodeById.get("missing-a").x));
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
