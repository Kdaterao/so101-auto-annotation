# JEPA-based subtask segmentation background

## Current decision

The initial decision was to keep gripper opening/closing segmentation. The updated
decision is to prototype V-JEPA 2.1 segmentation now as part of the
[two-day dataset pipeline plan](dataset_pipeline_mvp_plan.md), retaining gripper
events as context and a baseline. That plan records the current model choices,
processing order, and validation schedule. This background note and the plan do
not themselves implement the new runtime or add model dependencies.

## Intended role

Use a pretrained video JEPA encoder only to split demonstrations into meaningful
subtasks **offline during dataset creation**. Keep a VLM for interpreting each
subtask and grounding its goals.

The deployed robot architecture is **VLM + VLA**: the VLM supplies task context
and grounded goals, and the VLA produces robot actions. JEPA will not run in the
robot harness, controller, or inference pipeline. Its role ends after dataset
segmentation; JEPA planning and latent subgoal control are outside this project.

The prospective pipeline is:

1. Extract temporal video features and propose contiguous subtask segments.
2. Use the task prompt and a VLM to interpret the segments and identify goals.
3. Ground and track goal points using the existing perception pipeline.
4. Review uncertain boundaries and grounding, then save clean videos and sidecars.
5. Train and evaluate the VLM and VLA using the resulting grounded demonstrations.

Visual segmentation may help identify phases without a gripper event, such as
transport followed by insertion. A visual change alone does not establish a
semantic subtask boundary. A goal point may also need an instruction, gripper
state, or orientation constraint to describe the desired outcome.

## Closest reference: Action100M

[Action100M, Section 3](https://arxiv.org/html/2601.10592v1#S3) uses:

- A pretrained V-JEPA 2 ViT-g encoder to produce video features.
- Sampling one frame out of every four raw frames, with overlapping windows of
  64 sampled frames and a stride of eight sampled frames.
- Spatial pooling and averaging of overlapping features.
- Hierarchical agglomerative clustering with Ward linkage and temporal
  connectivity, so only adjacent spans can merge.
- VLM captions and subsequent LLM aggregation to describe the resulting segments.

This offers a segmentation approach using pretrained features and clustering,
without first training a boundary detector. Its segments still need evaluation
for robot subtasks. Its duration thresholds should not be copied blindly:
the paper filters some annotation stages to segments longer than four seconds,
while useful robot subtasks can be much shorter.

The [official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2)
lists encoder sizes of approximately 300M parameters (ViT-L), 600M (ViT-H),
and 1B (ViT-g). Action100M uses ViT-g at 384-pixel resolution.
Smaller encoders can be evaluated as a compute/quality tradeoff, but should not
be assumed to reproduce its segmentation quality.

## Compute caveat

[Action100M](https://arxiv.org/html/2601.10592v1) reports approximately
1.3 million V100 GPU-hours for **segmentation plus captioning** and 300,000
H100/H200 GPU-hours for LLM aggregation over 1.2 million videos totaling
14.6 years of footage. It does not report JEPA-only segmentation cost separately.

Those totals are not a runtime estimate for this pipeline or an RTX A6000.
Video resolution, temporal window overlap, batch size, decoding, and activation
memory all affect cost; parameter count alone does not establish speed.

## Experiment before adoption

- Manually annotate boundaries in a representative held-out sample, including
  quick grasp/release reversals, transport, insertion, and occlusion.
- Compare current gripper splitting with JEPA clustering and a hybrid that uses
  visual features alongside gripper events. Use the same episodes for comparison.
- Measure boundary precision/recall with explicit timing tolerances, missed and
  spurious segments, and whether the resulting segments yield useful grounded goals.
- Benchmark encoder inference, decoding, and clustering separately on short
  SO-101 episodes using the A6000. Record seconds per episode, throughput,
  CPU RAM, peak VRAM, model variant, resolution, sampling rate, and batch size.
- Cache features for segmentation experiments rather than rerunning inference
  for every clustering setting. Inspect results in the subtask viewer.
- Adopt only if it improves boundary/goal quality with acceptable processing
  cost. Retain gripper splitting as the baseline and human review for uncertain cases.

