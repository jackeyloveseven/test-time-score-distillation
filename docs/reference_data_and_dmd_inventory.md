# Reference Data and DMD Environment Inventory

Status: server inventory audited on 2026-07-13.

This note separates usable style references from merely numerous reference
images, and records the existing DMD implementation that can be reused later.

## Style-Relevant Sources

### Hyperion reference set

```text
path: /home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty
image count: 40
role: controlled smoke, teacher audit, named style-family experiments
```

These images have strong and diverse palette, line, material, and mark-making
cues. They are the cleanest immediate source for the reference-KV audit. Forty
images are not enough for the final generalization claim.

### Hyperion generated style gallery

```text
path: /home/ubuntu/chengyanli/items/save/attention_20260426_090948/d_grand/ext_styleshot
image count: 106
filename pattern: sty<family>_kp<prompt>
role: diagnostics-only within-family augmentation
```

These are generated images, not independent real references. They can create
multiple image instances for the same named style family and are useful for
testing exact-reference versus family leakage. They must be labeled synthetic
and cannot carry the paper's real-data claim.

### MultiBanana style subset

```text
path: /mnt/local/external_benches/MultiBanana/data/style
style tasks: approximately 52
image files: 156
task structure:
  <id>_0 = Image 1, style reference
  <id>_1 = Image 2, content image
  <id>_generated = benchmark output
  <id>_prompt.txt = "Change the style of Image 2 to the style of Image 1"
role: independent cross-dataset reference test
```

Use `<id>_0` as the reference image. The source and generated images are useful
for benchmark context but are not training targets for the Klein field student.
This set should remain untouched until the V1/V2 design is frozen.

## Sources That Are Not Style Training Data

### Qirui balanced benchmark references

```text
path: /home/ubuntu/chengyanli/qiruisbenchmark/benchmark_images/benchmark_final_balanced/ref
count: 122
```

Visual inspection shows a mixture dominated by people, characters, and identity
references. It is useful for a future broader Think-with-Image transfer test,
not for the first style-only claim.

### Real 9k reference set

```text
path: /mnt/local/felix_real_images/real_p4_10k_20260609/refs
count: 9,340
size: approximately 28 GiB
```

Visual inspection shows predominantly white-background person/identity
references. Raw count is large but the semantic task is different. Mixing it
into style training would silently turn the problem into identity preservation.

### MegaStyle cache

```text
path: /mnt/local/huggingface_cache/hub/datasets--tencent--MegaStyle-1.4M
local state: repository ref only; image payload is not downloaded
```

This is a candidate for the final real-data scale-up, but it is not currently a
local training source. Any download/curation step must preserve licenses,
family labels, hashes, and a frozen cross-dataset test.

## Immediate Split Ladder

The next experiments should increase evidence in stages rather than pretending
the 40-image source is already paper scale.

```text
Stage 0 teacher/conditioning audit:
  Hyperion 8 references x 3 prompts x 3 timesteps x 2 reference-noise views

Stage 1 V1 field smoke:
  Hyperion references, exact-image grouped split
  synthetic gallery only for explicit seen-family/exact-image diagnostics

Stage 2 V2 decision:
  same immutable Hyperion split and wrong-reference controls

Stage 3 independent test:
  untouched MultiBanana Image-1 references

Stage 4 paper scale:
  curated real style dataset with family labels and duplicate filtering
  MultiBanana remains cross-dataset test
```

Required split keys:

```text
reference_sha256
near_duplicate_cluster_id
style_family_id
source_dataset
synthetic_or_real
prompt/content group
```

## Existing DMD2 Environment

An existing DiffSynth DMD2 implementation is available:

```text
code: /mnt/local/yzh/prog/DiffSynth-Studio/diffsynth/diffusion/dmd2
environment: yzh-dmd
smoke dataset: /mnt/local/yzh/dataset/test-dmd
entrypoint: python -m diffsynth.diffusion.dmd2.run
```

Relevant components:

```text
loss.py              target/fake distribution matching surrogate
guidance_backend.py  standalone or LoRA fake-score model
training_module.py   alternating generator/guidance turns
runner.py            dual optimizer loop
adapter.py           pipeline-specific parameterization conversion
```

This code is reusable infrastructure, not a drop-in proof that the proposed
method is DMD. The current project must first provide:

1. a differentiable student sampling path;
2. a decoupled image-conditioned target score/velocity evaluated on generated
   noisy samples;
3. a dynamically trained fake-distribution score conditioned on the same
   prompt and reference;
4. a verified conversion between Klein9B velocity and the score quantity used
   by the DMD loss.

Before those four conditions hold, the correct stage name is on-policy field
aggregation. The existing DMD2 runner should be adapted only after the measured
student-trajectory field gap remains material.

## Current Decision

Quantity is not the bottleneck. Task purity and split validity are. The next
data action is to build the small immutable conditioning audit, not to ingest
the 9k identity set. The final large-scale source should contain true style or
appearance references and explicit family information.

