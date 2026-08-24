# Trace Observatory runbook

The Trace Observatory is a CPU-only, localhost-bound viewer for the seven independent Qwen
compact traces. It serves prebuilt safe JSON and a small persistent workspace; the serving process
does not load a model, import Torch or Transformers, or unpickle trace data.

## Current deployment roots

```text
Source traces (read-only):
/scratch/general/vast/u1653998/circuits/results/bonafide/raw-graph-observatory-qwen-selected-v1

Rebuildable viewer bundle:
/scratch/general/vast/u1653998/circuits/results/bonafide/raw-graph-observatory-viewer-v1

Persistent workspace state:
/uufs/chpc.utah.edu/common/home/u1653998/projects/circuits-observatory-state
```

The current bundle contains positions `65, 88, 120, 135, 162, 181, 184` and has bundle content
hash `6208b5a312857ee204d2a76da85a4b6c284a08a77062fb1197422b2b598f2731`. The replacement made
during final validation preserved the prior bundle at:

```text
/scratch/general/vast/u1653998/circuits/results/bonafide/raw-graph-observatory-viewer-v1.backup-20260823T223614Z-6208b5a3
```

The bundle is rebuildable; the workspace state is not. VAST remains subject to the CHPC inactivity
purge policy, so this viewer does not make the source traces archival.

## Rebuild the safe viewer bundle

Run synchronization in a development allocation because it opens the trusted compact artifacts
and loads the pinned offline tokenizer:

```bash
cd /uufs/chpc.utah.edu/common/home/u1653998/projects/worktrees/circuits-inspect-traced-graphs
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh

"$UV_PROJECT_ENVIRONMENT/bin/python" -m circuits.observatory sync \
  --trace-root /scratch/general/vast/u1653998/circuits/results/bonafide/raw-graph-observatory-qwen-selected-v1 \
  --site-root /scratch/general/vast/u1653998/circuits/results/bonafide/raw-graph-observatory-viewer-v1 \
  --state-root /uufs/chpc.utah.edu/common/home/u1653998/projects/circuits-observatory-state \
  --replace
```

Without `--replace`, synchronization fails if the destination exists. Replacement is atomic and
moves the old bundle to a timestamped sibling backup. Synchronization verifies compact-artifact
hashes before and after loading, validates every target slice round trip, validates label bindings,
and never writes into the source trace tree.

## Start the viewer

The JSON-only server is light enough for a login node if current CHPC policy permits it. Otherwise
run the identical command in a small CPU Slurm allocation.

```bash
cd /uufs/chpc.utah.edu/common/home/u1653998/projects/worktrees/circuits-inspect-traced-graphs
module load python/3.12.12 uv/0.11.14
source scripts/chpc_env.sh

"$UV_PROJECT_ENVIRONMENT/bin/python" -m circuits.observatory serve \
  --site-root /scratch/general/vast/u1653998/circuits/results/bonafide/raw-graph-observatory-viewer-v1 \
  --state-root /uufs/chpc.utah.edu/common/home/u1653998/projects/circuits-observatory-state \
  --host 127.0.0.1 \
  --port 8032
```

Startup validates the bundle manifest, canonical JSON hashes, trace/model/source bindings, and all
synthetic label occurrence/basis bindings. It fails closed if the bundle or state root is invalid
or unwritable. Stop it with `Ctrl-C`.

For a service that survives the launching shell, submit the lightweight scheduler-owned launcher
from the repository root:

```bash
sbatch scripts/bonafide/serve_trace_observatory.sbatch
squeue --name trace-observatory --me -o '%.18i %.8T %.20R'
```

Use the running job's node from the last column as the worker in the SSH command below. The default
job requests one CPU, 1 GiB, and four hours; override the walltime at submission if needed. Stop a
scheduler-owned viewer with `scancel JOB_ID`. The launcher remains localhost-only and accepts
optional `SITE_ROOT`, `STATE_ROOT`, and `PORT` exports; it refuses a non-loopback `HOST`.

## Forward the port

If the server runs on a CHPC login host, run this on the local computer:

```bash
ssh -N -L 8032:127.0.0.1:8032 u1653998@notchpeak.chpc.utah.edu
```

If the server runs on a worker such as `grn052`, tunnel through the login host to that worker so
the worker-local loopback binding remains private:

```bash
ssh -N \
  -J u1653998@notchpeak.chpc.utah.edu \
  -L 8032:127.0.0.1:8032 \
  u1653998@grn052
```

Then browse `http://127.0.0.1:8032`.

## Viewer semantics

- Each target row loads one independent trace; the browser never unions their topology.
- The default 100-edge graph is a target-connected upstream display projection. `Input nodes`
  defaults to `Hidden`; `Shown` restores them in ascending token-position order in a compact,
  wrapped bottom ribbon without changing the graph width. Fit includes the ribbon header and first
  two rows so a long input does not shrink the internal graph; pan or focus reaches later rows.
- Output is always the top band, displayed transformer layers are strictly ordered from highest to
  lowest, and the optional input-node ribbon is last. Connectivity influences horizontal ordering
  only; it never moves a neuron into another vertical layer.
- Input-attribution and output-contribution profiles in `Neuron evidence` retain every finite value
  stored for that neuron regardless of whether explicit input nodes are visible. They open on the
  strongest 18 values; `Show all` reveals the complete profile without changing the graph.
- Full source and displayed counts plus actual retained eligible edge-attribution mass remain
  visible in the provenance drawer.
- Edge attribution and weight, node attribution and activation, attribution sign and activation
  polarity, and occurrence and signed-basis identity remain separate.
- The two installed label sets are synthetic fixtures only. Their warning is deliberate; they
  prove the A/B overlay boundary and have no semantic meaning.
- Saved notes and pins are workspace state, not evidence artifacts. A revision conflict prevents
  silent overwrite from another tab.

The graph warning is literal: this is a pruned, local attribution approximation for one selected
logit, not a complete transcript of model computation or a faithfulness verdict.
