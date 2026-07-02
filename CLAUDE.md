# CLAUDE.md

This file is the operating contract for future agents working in this repository.
It is based on the current source tree, the current shell scripts, and the
project-local `RecogDrive_dataflowAndDataAdaptation.pdf` extracted on
2026-05-29. Treat code as the source of truth when this guide conflicts with
older documentation.

This worktree is on branch `recogdrive_qwen3vl_new_hdmap_all`. The runnable
Mogo + Qwen3 full-chain entry script names are unchanged and still use
single-view/single-frame Qwen3 image input by default. The active scripts now
override the yaml defaults to enable HDMap vector context, object vector context,
and `vlm_ablation=zero`; the model code also contains optional
multi-view/multi-frame Qwen3 inputs and GRPO behavior-cloning anchor knobs.
The reference pre-Qwen3 tree lives at
`/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive_ori`; diff against it when
you need to understand the original VLM migration. See the dedicated sections
"Qwen3-VL Migration" and "Optional Model Inputs" below before editing any VLM,
cache, or planner code.

## Non-Negotiable Behavior

- Be objective. Do not agree with an assumption because it came from the user.
  Verify it in code, configs, logs, or data.
- 回答用户时，始终使用中文。
- Do not revert or overwrite unrelated local changes. This repo may have a dirty
  worktree with active experiment edits.
- Prefer small, scoped patches. This codebase mixes upstream NAVSIM, InternVL,
  and local ReCogDrive changes; broad cleanup can silently break experiments.
- Before changing training, evaluation, cache, or data-generation paths, inspect
  the relevant script and config in the current working tree. Many scripts carry
  local absolute paths.
- Do not invent dataset fields. For new data adapters, map fields explicitly to
  `navsim.common.dataclasses` and document any missing information.
- Never assume online map access during IL/RL training. Maps enter mainly
  through metric caching, hidden-state/feature caching, PDM scoring,
  visualization, and scenario construction. The active chain reads cached HDMap
  vectors because `agent.use_hdmap_vector=true`; without a `Scene.map_api`, the
  non-strict HDMap builder returns all-zero map tokens.

## Project Shape

ReCogDrive is a Vision-Language-Action driving stack built on NAVSIM/nuPlan. The
main planning path combines:

1. An autoregressive VLM trained on driving QA and trajectory prompts.
2. Cached VLM hidden states from the current front-camera prompt.
3. Optional structured context tokens from HDMap lane/lane-connector vectors and
   current-frame object annotations.
4. A DiT-style diffusion planner that predicts an 8-step, 4-second ego
   trajectory in continuous `(x, y, heading)`.
5. Optional DiffGRPO/RL, where rewards are PDM scores computed from metric
   cache.

Important directories:

- `navsim/`: NAVSIM fork plus ReCogDrive agents, training, metric caching,
  scenario, PDM simulation/scoring, and visualization.
- `navsim/agents/recogdrive/`: main ReCogDrive agent, VLM backbone wrapper,
  feature builder, diffusion planner, DiT blocks, and utilities.
- `internvl_chat/`: InternVL training/inference stack used for Stage 1 VLM SFT.
- `scripts/`: runnable cache, training, evaluation, visualization, and dataset
  generation scripts.
- `docs/`: installation and training/evaluation overview.
- `vqa_evaluation/`: DriveLM, LingoQA, and DriveBench-style VQA evaluation code.
- `RecogDrive_dataflowAndDataAdaptation.pdf`: local data-flow and data-adapter
  note. It contains useful schema/map notes, but some model dimensions are stale
  relative to current source.

## Current Mogo Workflow

The active project workflow is Mogo + Qwen3-VL. Use these entry scripts as the
source of truth; many values are local absolute paths and are intentionally not
hidden behind generic docs.

1. Produce NAVSIM-style Mogo pkl/sensor data in the external repository:
   `/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_mogo/mogo-dataset-tools/4_general_recog_data.sh`
   on branch `rl_recogdrive`.
2. Produce Mogo metric cache in this repository:
   `scripts/cache_dataset/run_mogo_metric_caching.sh`.
3. Produce Qwen3 hidden-state cache:
   `scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh`.
4. Run imitation learning:
   `scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh`.
5. Run DiffGRPO/RL:
   `scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh`.
6. Run PDM scoring:
   `scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh`.
7. Visualize PDM CSV output:
   `scripts/visualization/plt_csv_all.sh`.
8. Render token-centered videos:
   `scripts/vis_scenario_video.py`.

Do not substitute the older InternVL / nuPlan entry scripts unless the user
explicitly asks for the legacy flow. The current Mogo Qwen3 cache filename is
`qwen3_feature.gz` for the default single-view/single-frame path, and the
planner expects `agent.vl_hidden_dim=2048`.

The entry scripts above are still single-view/single-frame for Qwen3 image
selection: `view_list=['cam_f0']`, `num_frames=1`, and
`use_video_input=false`. Do not confuse that with the yaml defaults: the active
full-chain scripts explicitly set `agent.use_hdmap_vector=true`, export
`USE_OBJECT_VECTOR=true`, and set `agent.vlm_ablation=zero` for IL/RL/eval.
Keep hidden-state cache, IL, RL, and online evaluation/video rendering aligned
on view/frame, HDMap, object, and VLM-ablation settings. Otherwise cache
filenames, required feature files, or checkpoint shapes will diverge.

### External Mogo PKL Production

The pkl production script lives outside this repository:

```bash
cd /pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_mogo
git checkout rl_recogdrive
bash mogo-dataset-tools/4_general_recog_data.sh
```

Current defaults in `4_general_recog_data.sh`:

- `MODE=worldsim` by default. It reads
  `/pcpt/pcpt/data/VLA/MOGO/rosbags/worldsim` and processes
  `WORLDSIM_CITIES="beijing shanghai"`.
- Worldsim output goes to
  `/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/<city>/dataset`.
  Each city writes `navsim_logs/train` and `sensor_blobs/train`, then symlinks
  them into sibling `trainval` directories.
- `MODE=legacy` reads `/pcpt/pcpt/data/VLA/MOGO/rosbags`, writes to
  `/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset`, uses `0312-2` as val, and
  uses `0313-1 0316 0317 0318 0319` as train by default.
- The generated `trainval` directory is what the current project scripts read.

### Mogo Metric Cache

Run:

```bash
bash scripts/cache_dataset/run_mogo_metric_caching.sh
```

Current script defaults:

- `OPENSCENE_DATA_ROOT=/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset`
- `NAVSIM_EXP_ROOT=/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp`
- `CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_mogo_v3`
- `train_test_split=trainval`
- The shell script relies on
  `navsim/planning/script/config/metric_caching/default_mogo_metric_caching.yaml`
  for `mogo_map.sqlite_path`, currently
  `$OPENSCENE_DATA_ROOT/maps/hm_beijing_3.4.9_20260521.sqlite`.
- `mogo_map.utm_zone=50`

`run_mogo_metric_caching.py` uses the same default yaml above, which sets a
dense 10Hz Mogo metric window (`num_history_frames=16`,
`num_future_frames=50`, `frame_interval=5`) and injects the SQLite map adapter.
Metric cache is required by RL rewards, PDM evaluation, and visualization.

### Qwen3 Hidden-State Cache

Run:

```bash
bash scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh
```

Current defaults:

- `DATA_ROOT=/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319`
- `TRAIN_TEST_SPLIT=mogo_trainval`
- `QWEN3_VL_PATH=/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct`
- `CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_qwen3_notrain_hdmap_cache`
- `SQLITE_PATH=$OPENSCENE_DATA_ROOT/maps/hm_beijing_3.4.9_20260521.sqlite`
- `MOGO_UTM_ZONE=50`
- `GPUS=4`, `MASTER_PORT=9696`
- `USE_OBJECT_VECTOR=true` by default.
- Python entry:
  `navsim/planning/script/run_dataset_caching_multi_node.py`
- Key Hydra overrides: `agent.cache_hidden_state=True`, `agent.cache_mode=True`,
  `agent.vlm_type=qwen3`, `agent.vl_hidden_dim=2048`,
  `agent.cam_type=single`, `agent.dit_type=small`, `agent.vlm_size=small`,
  `agent.use_hdmap_vector=true`, `mogo_map.sqlite_path=$SQLITE_PATH`,
  `mogo_map.utm_zone=$MOGO_UTM_ZONE`, and object-vector overrides when
  `USE_OBJECT_VECTOR=true`.

This stage writes `qwen3_feature.gz` and `trajectory_target.gz` under
`cache_path/log_name/token/` for the default single-view/single-frame path, and
also writes `hdmap_vector_feature.gz` plus `object_vector_feature_v1.gz` in the
active HDMap/object configuration.
If `view_list`, `num_frames`, or `use_video_input` changes,
`ReCogDriveFeatureBuilder.get_unique_name()` writes
`qwen3_feature_v{n_views}_f{num_frames}[_vid].gz` instead; re-cache and keep
the same overrides in IL/RL/eval. Re-run cache whenever `QWEN3_VL_PATH`, prompt
logic, selected hidden-state layers, view/frame settings, HDMap/object settings,
or the Mogo pkl content changes.

### IL Training

Run:

```bash
bash scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh
```

Current defaults:

- Same `DATA_ROOT`, `TRAIN_TEST_SPLIT`, `QWEN3_VL_PATH`, and hidden-state
  `CACHE_PATH` as the Qwen3 cache script.
- `GPUS=4`, `BATCH_SIZE=80`, `MAX_EPOCHS=200`.
- `USE_OBJECT_VECTOR=true` by default.
- Python entry: `navsim/planning/script/run_training_recogdrive.py`.
- Key Hydra overrides: `agent.grpo=False`, `agent.cache_hidden_state=True`,
  `agent.vlm_type=qwen3`, `agent.vl_hidden_dim=2048`,
  `agent.sampling_method=ddim`, `agent.use_hdmap_vector=true`,
  `agent.vlm_ablation=zero`, object-vector overrides when enabled,
  `use_cache_without_dataset=True`, `force_cache_computation=False`,
  and `+log_split@_global_=mogo`.
- Experiment name: `training_recogdrive_qwen3_obj_vlmzero_IL`.

Because `use_cache_without_dataset=True`, IL training reads cached features and
does not load Qwen3 online. Note the script currently calls
`/pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all` after training.

### RL Training

Run:

```bash
bash scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh
```

Current defaults:

- `DATA_ROOT=/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319`
- `TRAIN_TEST_SPLIT=mogo_trainval`
- `QWEN3_VL_PATH=/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct`
- `CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_qwen3_notrain_obj_cache`
- `METRIC_CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_mogo_v3`
- `CHECKPOINT=$NAVSIM_EXP_ROOT/training_recogdrive_qwen3_obj_vlmzero_IL/2026.06.30.14.39.02/lightning_logs/version_0/checkpoints/epoch=197-step=5346.ckpt`
- `GPUS=4`, `dataloader.params.batch_size=16`,
  `trainer.params.max_epochs=20`
- `USE_OBJECT_VECTOR=true` by default.
- Python entry: `navsim/planning/script/run_training_recogdrive_rl.py`.
- Key Hydra overrides: `agent.grpo=True`, `agent.cache_hidden_state=True`,
  `agent.vlm_type=qwen3`, `agent.vl_hidden_dim=2048`,
  `agent.sampling_method=ddim`, `agent.use_hdmap_vector=true`,
  `agent.vlm_ablation=zero`, `agent.metric_cache_path=$METRIC_CACHE_PATH`,
  `agent.reference_policy_checkpoint=$CHECKPOINT`,
  object-vector overrides when enabled, `use_cache_without_dataset=True`,
  `train_logs=null`, `val_logs=null`, and
  `train_token_whitelist_path=navsim/local_config/data_selection/rl_grpo_n4000_selected_tokens_v1.txt`.
- stdout/stderr are redirected to `train_recogdrive_qwen3_agent_rl_2b.txt`.

The RL checkpoint and reference policy checkpoint must come from a Qwen3 IL run
with matching 2048-wide VLM projectors, HDMap/object encoders, and
`vlm_ablation` setting. The script's `CACHE_PATH` default currently differs
from the cache/IL script default; align it before long runs unless that
separate object-cache directory has already been populated. This script also
calls `simple_gpu_stress.py all` after training.

### PDM Evaluation

Run:

```bash
bash scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh
```

Current defaults:

- `DATA_ROOT=/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319`
- `TRAIN_TEST_SPLIT=mogo_trainval`, with explicit
  `train_test_split.data_split=val`.
- `CHECKPOINT=$NAVSIM_EXP_ROOT/training_recogdrive_qwen3_obj_vlmzero_RL/2026.07.01.03.42.32/lightning_logs/version_0/checkpoints/epoch=19-step=640.ckpt`
- `METRIC_CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_mogo_v2`
- `SQLITE_PATH=$OPENSCENE_DATA_ROOT/maps/hm_beijing_3.4.9_20260521.sqlite`
- `MOGO_UTM_ZONE=50`
- `GPUS=4`, `MASTER_PORT=9696`
- `USE_OBJECT_VECTOR=true` by default.
- Python entry: `navsim/planning/script/run_pdm_score_recogdrive.py`.
- Key Hydra overrides: `agent.cache_hidden_state=False`,
  `agent.vlm_type=qwen3`, `agent.vl_hidden_dim=2048`,
  `agent.sampling_method=ddim`, `agent.use_hdmap_vector=true`,
  `agent.vlm_ablation=zero`, object-vector overrides when enabled,
  `mogo_map.sqlite_path=$SQLITE_PATH`, `mogo_map.utm_zone=$MOGO_UTM_ZONE`,
  and `experiment_name=recogdrive_qwen3_hdmap_ddim_IL_eval`.

Evaluation runs the Qwen3 backbone online per sample because
`agent.cache_hidden_state=False`; it does not read the hidden-state cache, but
it does read the metric cache. The eval script still points at
`metric_cache_mogo_v2` while metric caching and RL default to
`metric_cache_mogo_v3`; align this before comparing scores.

### Visualization and Video

CSV visualization:

```bash
bash scripts/visualization/plt_csv_all.sh
```

Current `plt_csv_all.sh` is configured for the 0312_0319 Mogo dataset. It reads
a hard-coded PDM CSV, uses `scene_filter=mogo_10hz`, reads
`$NAVSIM_EXP_ROOT/metric_cache_mogo_v3`, and passes the Beijing SQLite map
`hm_beijing_3.4.9_20260521.sqlite`.

Token-centered video rendering:

```bash
python scripts/vis_scenario_video.py --model both --window 30 TOKEN...
```

`vis_scenario_video.py` defaults to worldsim Shanghai
`OPENSCENE_DATA_ROOT`/`NAVSIM_EXP_ROOT`, uses the local Qwen3-VL weights, and
has hard-coded IL/RL checkpoint paths in the repository `exp/` directory. It
loads the `mogo_trainval` scene filter, runs the agent online
(`cache_hidden_state=False`), auto-selects the first SQLite map under
`OPENSCENE_DATA_ROOT/maps` unless `--mogo-map-sqlite-path` is provided, and
defaults `--mogo-map-utm-zone=51`. Use `--full-log` for the whole log; otherwise
it renders a token-centered window.

### Train/Val Split for Mogo

Hidden-state caching and metric caching consume one split and write a flat
`cache_path/log_name/token/*.gz` layout. They do not create separate train and
val cache trees. The current model-facing Mogo split is
`train_test_split=mogo_trainval`, which resolves to `data_split=trainval` plus
`scene_filter=mogo_10hz`.

`mogo_10hz` adapts dense 10Hz pkl frames to the model's 2Hz cadence:
`num_history_frames=4`, `num_future_frames=10`, `frame_stride=5`, and
`frame_interval=5`.

Train/val separation happens at training time through top-level `train_logs` and
`val_logs`. The IL and RL scripts inject:

```bash
'+log_split@_global_=mogo'
```

This loads `navsim/planning/script/config/training/log_split/mogo.yaml`.
Regenerate that file from the on-disk `train/` and `val/` directories when the
pkl set changes:

```bash
python scripts/cache_dataset/gen_mogo_log_split.py \
  --logs-root /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset/navsim_logs
```

If `train_logs` and `val_logs` are both `null`, `CacheOnlyDataset` loads every
cached log for both datasets, so validation equals training.

### Optional Qwen3-VL SFT

The default Mogo workflow above consumes the original frozen Qwen3-VL weights.
The `internvl_chat/` trainer CANNOT fine-tune Qwen3-VL:
`internvl/train/internvl_chat_finetune.py` hardcodes
`InternVLChatModel`/`InternVLChatConfig`/`InternVisionModel` and only loads
InternVL-architecture weights.

To fine-tune the Qwen3 backbone, use the ms-swift LoRA flow:

- Train LoRA: `scripts/training/run_qwen3vl_sft_lora.sh`.
- Merge LoRA into a standalone model directory:
  `scripts/training/merge_qwen3vl_lora.sh`.
- Convert existing ReCogDrive JSONL to ms-swift format with
  `scripts/training/convert_recogdrive_jsonl_to_swift.py`.
- Point `QWEN3_VL_PATH` / `agent.vlm_path` at the merged model directory and
  re-run `run_caching_recogdrive_qwen3_hidden_state.sh`.

`RecogDriveBackbone` has no adapter path and loads Qwen3 with
`Qwen3VLForConditionalGeneration.from_pretrained(vlm_path)`, so merged weights
are required.

## Qwen3-VL Migration

This branch swaps the VLM backbone from InternVL to
**Qwen3-VL-2B-Instruct** (`model_type=qwen3_vl`, `text_config.hidden_size=2048`,
28 text layers, vision hidden 1024). Reference pre-swap tree:
`/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive_ori`.

The model weights live locally at two paths seen in the scripts:
`/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct` (cache/train
default) and `/pcpt/pcpt/project/lijun/.code/Qwen3-VL-2B-Instruct` (eval default
and the path hardcoded in the integration test). Both exist and carry identical
`config.json` (hidden 2048). Keep `agent.vlm_path` consistent across stages.

### Qwen3 script contract

The runnable Qwen3/Mogo order is documented in `Current Mogo Workflow` above.
Keep this lower migration section focused on model wiring, not on duplicating
entrypoint paths. The active Qwen3 contract is:

- Hidden-state cache and IL/RL training use `agent.cache_hidden_state=True`,
  `agent.vlm_type=qwen3`, and `agent.vl_hidden_dim=2048`.
- Hidden-state cache additionally sets `agent.cache_mode=True`, so the feature
  builder loads Qwen3 and writes the model-aware cache file. Default
  single-view/single-frame writes `qwen3_feature.gz`; multi-view/multi-frame
  writes `qwen3_feature_v{n_views}_f{num_frames}[_vid].gz`.
- IL/RL training use `use_cache_without_dataset=True` and read the cached Qwen3
  feature through `CacheOnlyDataset`; they do not run Qwen3 online. Cache, IL,
  RL, and eval/video must share `view_list`, `num_frames`, `use_video_input`,
  `use_hdmap_vector`, `use_object_vector`, and `vlm_ablation` when those
  options are enabled.
- PDM evaluation and video/CSV visualization run with
  `agent.cache_hidden_state=False`, so the agent loads Qwen3 online and calls
  `forward_from_paths(...)` per sample. Multi-view/multi-frame online eval uses
  `ReCogDriveAgent._encode_views(...)` and the same prompt helpers as cache
  mode.
- RL is now migrated through
  `scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh`; do not use
  the older `run_recogdrive_train_multi_node_rl_2b.sh` for the active Mogo Qwen3
  flow unless intentionally testing the legacy InternVL path.

### Verdict: is the Qwen3 swap wired correctly?

Checked code path "script flag -> hydra config -> code consumer". Findings:

- Config: every `agent.*` key the three Qwen3 scripts pass exists in
  `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
  (`vlm_type`, `vlm_path`, `vl_hidden_dim`, `cam_type`, `cache_hidden_state`,
  `cache_mode`, `dit_type`, `vlm_size`, `sampling_method`, `grpo`,
  `checkpoint_path`, `lr`, `view_list`, `num_frames`, `use_video_input`,
  `use_hdmap_vector`, `use_object_vector`, `vlm_ablation`, `bc_coeff`,
  `bc_teacher_deterministic`). No unknown-key hydra errors expected.
- Cache-name coupling is correct for the default path: `agent.get_feature_builders()`
  forwards `model_type=self.vlm_type`, and `get_unique_name()` returns
  `qwen3_feature` for Qwen3 single-view/single-frame. For multi-view/multi-frame
  Qwen3 it includes the view/frame tag. Cache and train must pass matching
  `vlm_type`, `view_list`, `num_frames`, and `use_video_input` so the
  read/write filenames match. Confirmed by
  `tests/test_qwen3_recogdrive_integration.py` for the base path.
- Dim plumbing is correct: scripts pass `agent.vl_hidden_dim=2048`, which is the
  real Qwen3-VL-2B `text_config.hidden_size`, and the planner's VLM projectors
  use it. `py_compile` clean on all core files; `transformers==4.57.1` in the
  active env exposes `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor` /
  `Qwen3VLVideoProcessor` (import-verified).
- Lightweight unit tests pass:
  `pytest tests/test_qwen3_recogdrive_integration.py -k "config or planner or
  select"` -> 5 passed. The remaining tests load the real model/processor or
  read Mogo pkl/JPG and need GPU + data on disk.

### Things to watch (not blockers, verify before a long run)

- `agent.vl_hidden_dim` must be passed explicitly. The yaml default is `1536`
  (not `null`), so the agent's `qwen3 -> 2048` auto-inference never triggers. If
  a future Qwen3 script forgets `agent.vl_hidden_dim=2048`, the planner builds
  1536-wide projectors and Qwen3's 2048-wide hidden states fail at the first
  `nn.Linear`.
- The Qwen3 prompt still contains a literal `<image>\n` in the text while the
  chat template inserts the real vision placeholders from the
  `{"type": "image"}` content entry. The literal token is processed as plain
  text, not as the image slot. This matches `recog_drive_ori` prompt wording but
  is worth confirming against Qwen3-VL token accounting if results look off.
- InternVL and Qwen3 caches are not interchangeable: `internvl_feature.gz` vs
  `qwen3_feature.gz`, and 1536 vs 2048 width. Never mix a cache built with one
  `vlm_type` into a run configured for the other.
- Checkpoints are not portable across VLMs. An IL checkpoint trained on InternVL
  hidden states has 1536-wide projectors; loading it into a Qwen3 (2048) agent
  silently drops those layers under `initialize()`'s `strict=False` shape
  filter. Eval `CHECKPOINT` must come from a Qwen3 IL run.
- `cache_mode` semantics: the feature builder only constructs a backbone when
  `cache_hidden_state and cache_mode` are both true (the caching script). In IL
  training `cache_mode` stays False, so no Qwen3 model is loaded during training
  - only the cache is read.
- Multi-view/multi-frame Qwen3 is zero-impact only when defaults stay at
  `view_list=['cam_f0']`, `num_frames=1`, and `use_video_input=false`. The
  Qwen3 backbone can treat frames as multiple images, or as `videos=` when
  `use_video_input=true` and `num_frames>1`; keep that choice consistent across
  cache and online eval.
- `agent.use_hdmap_vector=true` appends `HDMapVectorFeatureBuilder`. If
  `scene.map_api` is missing and `strict_hdmap_vector=false`, it returns zero
  map tokens; with `strict_hdmap_vector=true` it raises. PDM eval only passes a
  `Scene` when `agent.requires_scene` is true, and the current agent sets that
  from `use_object_vector`, not `use_hdmap_vector`; HDMap-only eval therefore
  needs a code/config check before relying on non-zero map tokens.
- `agent.use_object_vector=true` sets `requires_scene=True`, appends
  `ObjectVectorFeatureBuilder`, and creates `object_vector_feature_v1.gz` in
  cache mode. Missing that file makes `CacheOnlyDataset` drop samples, so rerun
  cache before training with object tokens.
- `agent.vlm_ablation=zero` zeros only VLM summary/context tokens after
  projection. Object tokens, HDMap tokens, ego status, action/history inputs,
  and GRPO reward logic remain active.

## Core ReCogDrive Files

- `navsim/agents/recogdrive/recogdrive_agent.py`
  - Implements `ReCogDriveAgent`.
  - Loads checkpoints in `initialize()` by filtering `agent.` prefixes.
  - Builds `ReCogDriveFeatureBuilder` and `TrajectoryTargetBuilder`, forwarding
    `model_type=self.vlm_type`, hidden-state layer indices, `view_list`,
    `num_frames`, `use_video_input`, and `video_fps` to the feature builder. It
    appends `HDMapVectorFeatureBuilder` and `ObjectVectorFeatureBuilder` only
    when their switches are enabled.
  - Uses `RecogDriveBackbone` only when hidden states are not already cached
    (`not cache_hidden_state and not cache_mode`), or via the feature builder in
    `cache_mode`.
  - Constructor args added beyond the original ReCogDrive path include
    `vl_hidden_dim`, `normalize_vl_hidden_states`, hidden-state layer indices,
    `vlm_ablation`, multi-view/multi-frame knobs (`view_list`, `num_frames`,
    `use_video_input`, `video_fps`), HDMap vector knobs, object vector knobs,
    and GRPO BC knobs (`bc_coeff`, `bc_teacher_deterministic`).
    `vl_hidden_dim`, structured-token switches, `vlm_ablation`, and BC knobs
    flow into `make_recogdrive_config(...)` and then the planner.
  - Auto-inference of `vl_hidden_dim` (`qwen3 -> 2048`, `large -> 3584`, else
    1536) only fires when the arg is `None`. The hydra default is the literal
    `1536`, **not** `null`, so that branch is effectively dead unless you pass
    `agent.vl_hidden_dim=null`. The Qwen3 scripts therefore pass
    `agent.vl_hidden_dim=2048` explicitly. Do not rely on auto-inference.
  - `compute_trajectory(...)` and `compute_trajectory_vis(...)` are scene-aware:
    if the caller passes `Scene` and a builder implements
    `compute_features_from_scene`, the agent uses that scene so HDMap/Object
    features can be built during eval.
  - Owns `action_head = ReCogDriveDiffusionPlanner(...)`.

- `navsim/agents/recogdrive/recogdrive_features.py`
  - Builds `history_trajectory`, `high_command_one_hot`, `status_feature`,
    optional image-path tensor, and optional VLM `first_hidden_state` /
    `last_hidden_state`.
  - Current source defines 13 status features:
    `cmd_0..cmd_3`, `vx`, `vy`, `ax`, `ay`, `yaw_rate`, `curvature`,
    `curvature_rate`, `steering_wheel_angle`, and `steering_wheel_rate`.
  - Do not regress this to the PDF's older 8-dimensional description or the
    previous branch's 10-dimensional state.
  - `get_unique_name()` is model-aware and view/frame-aware. InternVL returns
    `internvl_feature`. Qwen3 single-view/single-frame returns `qwen3_feature`;
    Qwen3 multi-view/multi-frame returns
    `qwen3_feature_v{n_views}_f{num_frames}[_vid]`. The same `agent.vlm_type`,
    `view_list`, `num_frames`, and `use_video_input` must be used wherever a
    cache is read or written. See "Qwen3-VL Migration".
  - `select_hidden_state(hidden_states, layer_index)` indexes the
    `hidden_states` tuple, honoring negative indices. `first_hidden_state_index`
    (default 0) and `last_hidden_state_index` (default -1) are threaded from the
    agent so the chosen layers are configurable.
  - Prompt construction now uses `command_code_to_text(...)` and shared
    `recogdrive_view_utils` helpers in both cache-mode feature building and
    online multi-view/multi-frame eval. Do not reintroduce hardcoded
    3-command prompt mappings.

- `navsim/agents/recogdrive/recogdrive_view_utils.py`
  - Single source of truth for multi-view/multi-frame image path selection and
    prompt wording. It turns `AgentInput.cameras` into `[view][frame]` paths
    with edge-padding and builds perception text that matches the selected
    view/frame mode.
  - `cam_f0` is the default. For Mogo multi-view, prefer real views such as
    `cam_f0`, `cam_l0`, `cam_r0`, `cam_l1`, `cam_r1`, and `cam_b0`; empty
    views are edge-padded and can silently duplicate a valid view.

- `navsim/agents/recogdrive/hdmap_vector_features.py` and
  `navsim/agents/recogdrive/hdmap_vector_encoder.py`
  - Optional `agent.use_hdmap_vector=true` path. The builder samples nearby
    lanes/lane-connectors from `Scene.map_api`, encodes centerline/left/right
    boundary points plus route, traffic-light, and speed-limit metadata, and
    writes `hdmap_vector_feature.gz`.
  - Non-strict mode returns all-zero map features when `scene.map_api` is
    missing or map lookup fails; strict mode raises. `MapEncoder` turns valid
    map polygons into DiT context tokens.

- `navsim/agents/recogdrive/object_vector_features.py`
  - Optional `agent.use_object_vector=true` path. The agent then sets
    `requires_scene=True`, extracts current-frame GT annotations from `Scene`,
    and writes `object_vector_feature_v1.gz`.
  - Object continuous features are 15-D (`x/y/z`, box size, sin/cos heading,
    velocity, speed, distance, bearing, TTC proxy), with 5 type ids
    (`vehicle`, `pedestrian`, `bicycle`, `static`, `unknown`). Object tokens are
    sorted by a simple risk score and padded/truncated to `max_obj_tokens`.

- `navsim/agents/recogdrive/recogdrive_backbone.py`
  - `RecogDriveBackbone` now supports three `model_type` values: `internvl`,
    `qwen` (Qwen2.5-VL, partially wired), and `qwen3` (Qwen3-VL, the active
    backbone on this branch).
  - InternVL path uses `AutoModel.from_pretrained(..., trust_remote_code=True,
    use_flash_attn=True, torch_dtype=torch.bfloat16)` and injects image context
    tokens into an `internvl2_5` conversation template via `forward(...)`.
  - Qwen3 path loads `Qwen3VLForConditionalGeneration` plus a processor built by
    `load_qwen3_processor(...)` (falls back to hand-assembling
    `Qwen3VLProcessor` from `Qwen2Tokenizer` + image/video processors +
    `chat_template.json` when `AutoProcessor` raises the local fast-tokenizer
    `model_type` error). Inference goes through `forward_from_paths(...)`, which
    accepts `[batch][view][frame]` path matrices. Single-view/single-frame
    preserves the previous chat-template call; multi-view/multi-frame uses
    `_forward_qwen3(...)` with multiple image placeholders, or
    `_forward_qwen3_video(...)` with `videos=` when `use_video_input=true` and
    `num_frames>1`. All Qwen3 paths call the model with
    `output_hidden_states=True`.
  - All three import their HF classes at the top of the file now; the unused
    `transformers` imports in the `qwen` branch are no longer the gap the PDF
    era described.
  - Both InternVL and Qwen3 return `CausalLMOutputWithPast`-style outputs whose
    `hidden_states` tuple is consumed by the agent / feature builder.
  - `self.vl_hidden_dim` is read from the loaded model config (InternVL
    `llm_config.hidden_size`, Qwen `text_config.hidden_size`). For
    Qwen3-VL-2B-Instruct this is **2048**.

- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
  - Implements `ReCogDriveDiffusionPlanner`.
  - Supports `ddim`, `ddpm`, and `flow` sampling.
  - Projects first/last VLM hidden states into context tokens, encodes ego
    status, encodes noisy action tokens, runs LightningDiT, decodes trajectory
    noise/velocity, and denormalizes to ego-frame trajectory.
  - VLM input width is now driven by `config.vl_hidden_dim` (the old hardcoded
    `3584` if `vlm_size=="large"` else `1536` was removed). The three VLM
    projectors (`vl_summary_projector`, `vl_context_first_projector`,
    `vl_context_last_projector`) are all `nn.Linear(vl_hidden_dim,
    input_embedding_dim)`, so a wrong `vl_hidden_dim` is a shape mismatch at
    load/forward. For Qwen3 it must be 2048.
  - Optional `normalize_vl_hidden_states` inserts `LayerNorm(vl_hidden_dim)`
    before each projector (`vl_summary_norm`, `vl_context_first_norm`,
    `vl_context_last_norm`); default is `nn.Identity`. Enabling it adds new
    parameters, so a checkpoint trained with one setting will partially skip
    these layers under the agent's `strict=False` load.
  - `build_context_tokens(...)` always creates VLM summary/context tokens. When
    enabled, it appends HDMap tokens and object tokens after VLM context tokens
    and returns a key-padding mask where VLM tokens are always valid and
    structured padding slots are invalid.
  - `vl_summary_tokens.mean(1)` pools over the (padded) sequence dimension to
    build the summary token; padded hidden states are zero, so they bias the
    mean slightly. This remains true for longer multi-view/multi-frame hidden
    sequences.
  - `config.vlm_ablation == "zero"` zeros projected VLM summary/context tokens
    inside `encode_vl_*`; HDMap/Object tokens and non-VLM inputs remain active.
  - `grpo=True` initializes `MetricCacheLoader`, `PDMSimulator`, `PDMScorer`,
    loads an IL reference checkpoint, creates frozen `old_policy`, and uses
    `bc_coeff` / `bc_teacher_deterministic` from `GRPOConfig` for the optional
    behavior-cloning anchor.

- `navsim/agents/recogdrive/recogdrive_dit.py`
  - LightningDiT backbone.
  - 16 transformer blocks; when `interleave_attention=True`, even layers use
    action self-attention and odd layers use cross-attention to VLM context.

- `navsim/planning/training/agent_lightning_module.py`
  - `AgentLightningModule` handles ordinary training.
  - `AgentLightningDiT` handles GRPO-style logging of `loss`, `reward`,
    `policy_loss`, and `bc_loss`.

## Optional Model Inputs

The yaml defaults keep these capabilities off, but the current single
full-chain scripts explicitly enable HDMap vectors, object vectors, and
`vlm_ablation=zero`. The scripts remain single-view/single-frame Qwen3 unless
view/frame overrides are provided. Relevant knobs:

- `view_list`, `num_frames`, `use_video_input`, `video_fps`: control Qwen3 image
  selection. Cache mode and online eval share `recogdrive_view_utils`; the
  single-view/single-frame path keeps the historical prompt and cache name.
- `use_hdmap_vector`, `max_map_tokens`, `map_sample_points`, `hdmap_radius`,
  `strict_hdmap_vector`: add structured HDMap context tokens. This path needs a
  real `Scene.map_api` for non-zero map tokens and writes
  `hdmap_vector_feature.gz` during cache generation.
- `use_object_vector`, `max_obj_tokens`, `obj_range_x`, `obj_range_y`,
  `include_static_objects`, `object_token_dropout`: add current-frame object
  context tokens from `Scene` annotations. This path writes
  `object_vector_feature_v1.gz` and requires scene-aware cache/eval.
- `vlm_ablation=zero`: zeros only VLM projected tokens. Use it for ablation, not
  as a substitute for disabling object/map inputs.

Enable optional inputs consistently across hidden-state cache, IL, RL, and
online eval. Do not train from a cache produced with a different view/frame,
structured-token, or `vlm_ablation` setting.

## Model Dimensions and Current Gotchas

- Trajectory target is always `(8, 3)` for 4 seconds at 0.5 second intervals.
- `small` DiT: `num_heads=8`, `head_dim=48`, inner dim 384, output dim 512.
- `large` DiT: `num_heads=32`, `head_dim=48`, inner dim 1536, output dim 1536.
- VLM hidden dim is now an explicit knob: `agent.vl_hidden_dim`. InternVL small
  = 1536, InternVL/`vlm_size=large` = 3584, **Qwen3-VL-2B = 2048**. The planner
  reads `config.vl_hidden_dim`; the old `vlm_size`-only branch was removed.
- Current planner config has `ego_status_dim=13`.
- The PDF describes older `feature_encoder`, `his_traj_encoder`, and 8-D state
  assembly. Current source uses `vl_summary_projector`,
  `vl_context_first_projector`, `vl_context_last_projector`, gated context
  fusion, `ego_status_encoder`, `action_encoder`, and `fusion_projector`.
- Prompt construction uses `command_code_to_text(...)` for the 4-D Mogo command
  vector. Before editing command handling, inspect actual `driving_command`
  arrays in the target logs and keep cache/eval prompt helpers aligned.
- Variable-length VLM hidden states are batched with
  `torch.nn.utils.rnn.pad_sequence(..., batch_first=True, padding_value=0.0)` in
  every `custom_collate_fn` (`run_training_recogdrive.py`,
  `run_training_recogdrive_rl.py`, `_ema`, `_accelerate`). Qwen3-VL emits a
  variable visual-token count per image, so this padding is what makes
  per-sample sequence-length differences safe. `history_trajectory`,
  `high_command_one_hot`, `status_feature`, and `trajectory` are plain
  `torch.stack`, so those must stay fixed-shape.
- `RecogDriveBackbone` now imports its HF classes at module top
  (`Qwen2_5_VLForConditionalGeneration`, `Qwen3VLForConditionalGeneration`,
  `Qwen3VLProcessor`, `Qwen3VLVideoProcessor`, `Qwen2Tokenizer`,
  `AutoImageProcessor`). The old "Qwen branch references undefined classes"
  gotcha is fixed. InternVL and Qwen3 are both maintained paths now.

## Data Loading

NAVSIM logs are pickle files under:

```text
${OPENSCENE_DATA_ROOT}/navsim_logs/${train_test_split.data_split}
```

Sensor blobs are under:

```text
${OPENSCENE_DATA_ROOT}/sensor_blobs/${train_test_split.data_split}
```

Main classes:

- `navsim.common.dataloader.SceneLoader`
  - Loads and filters log pickle files.
  - Splits scenes by `SceneFilter.num_history_frames`,
    `num_future_frames`, and `frame_interval`.
  - Filters by `has_route`, log names, optional tokens, and max scenes.
  - Builds token-to-log mappings for distributed evaluation/caching.

- `navsim.common.dataclasses.Scene`
  - Wraps frames, metadata, map API, annotations, sensors, and ego state.
  - `get_agent_input()` returns only history/current inputs.
  - `get_future_trajectory()` returns local ego-frame future poses.

- `navsim.planning.training.dataset.Dataset`
  - Normal training/cache dataset.
  - Cache layout is `cache_path/log_name/token/{builder_name}.gz`.

- `CacheOnlyDataset`
  - Loads already cached feature/target files without building `SceneLoader`.
  - Supports optional `tokens` filtering, used by current RL token whitelist.

- `Dataset_For_Traj`
  - Used by `run_generate_dataset.py` to emit trajectory QA JSONL.

- `Dataset_For_Pipeline`
  - Used by `run_generate_dataset_pipeline.py`.
  - Extracts front-camera-FOV objects, 3D boxes, labels, names, future
    velocities, and 2D boxes for QA generation.

## Required NAVSIM PKL Fields

For data adaptation, provide these fields in each frame dictionary:

- Required core: `token`, `timestamp`, `log_name`, `map_location`,
  `ego2global_translation`, `ego2global_rotation`, `ego_dynamic_state`,
  `driving_command`, `roadblock_ids`, `traffic_lights`, `anns`.
- Required annotations:
  - `anns["gt_boxes"]`: `(N, 7)` as `(x, y, z, length, width, height, heading)`.
  - `anns["gt_names"]`: class names such as `vehicle`, `pedestrian`,
    `bicycle`.
  - `anns["gt_velocity_3d"]`: `(N, 3)`.
  - `anns["track_tokens"]`: temporal object ids.
  - `anns["instance_tokens"]`: instance ids; less central but expected by
    dataclass construction.
- Required for camera work: `cams`, each selected camera's `data_path`,
  `cam_intrinsic`, `sensor2lidar_rotation`, `sensor2lidar_translation`,
  and `distortion`.
- Required for lidar work only if lidar is loaded: `lidar_path`.
- Optional or currently not central in main flow: `sample_prev`, `sample_next`,
  `scene_name`, `log_token`, `vehicle_name`, `ego2global`, `lidar2ego`,
  `lidar2global`, `frame_idx`, `flow_gt_final_path`, `occ_gt_final_path`.
- `can_bus` is used opportunistically for yaw rate if index 15 exists;
  otherwise yaw rate defaults to 0.0.

MOGO-style adapter mapping from the PDF:

- `anns["gt_boxes"]`: `position` -> `(x, y, z)`, `bbox_size` ->
  `(length, width, height)`, `yaw` -> `heading`.
- `anns["gt_names"]`: `class`.
- `anns["gt_velocity_3d"]`: velocity.
- `anns["track_tokens"]`: `track_id`.

## Maps and Metric Cache

Map version is effectively fixed to `nuplan-maps-v1.0` in current code. Set:

```bash
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps-v1.0
```

Map-related log fields:

- `map_location`: selects the nuPlan map via `get_maps_api`.
- `roadblock_ids`: route roadblocks.
- `traffic_lights`: lane connector light states.

The raw `map.gpkg` is not passed to the ReCogDrive neural model. It is used by:

- metric caching,
- PDM scoring,
- PDM closed planner/scenario construction,
- BEV visualization.

`MetricCache` stores:

- `trajectory`: PDM-Closed reference trajectory.
- `ego_state`: initial nuPlan ego state.
- `observation`: interpolated PDM observation.
- `centerline`: route centerline as `PDMPath`.
- `route_lane_ids`: lane ids used by PDM.
- `drivable_area_map`: `PDMDrivableMap`.

Metric cache generation reads maps; GRPO training and PDM evaluation read the
cache instead of directly reading `map.gpkg`.

Mogo / no-map caveat (introduced alongside the Qwen3 commit in
`navsim/common/dataclasses.py`):

- `Scene._build_map_api("las_vegas")` returns `None`. Mogo conversion uses
  `las_vegas` as a placeholder `map_name`; the hidden-state / trajectory
  training cache does not need a map api, and metric caching injects a Mogo map
  adapter explicitly. So `scene.map_api is None` is expected for Mogo training
  caches and must not be treated as an error.
- `Lidar.from_paths(...)` now returns an empty `Lidar()` when the lidar path is
  blank or resolves to a directory, so Mogo logs without point clouds still
  build. The integration test `test_mogo_scene_can_be_built_without_nuplan_map`
  asserts both behaviors.

## Data Generation

Trajectory JSONL:

```bash
sh scripts/generate_dataset/generate_internvl_dataset.sh
```

Python entry:

```bash
python navsim/planning/script/run_generate_dataset.py agent=recogdrive_agent train_test_split=navtrain
```

Output in the current script is `navsim_traj_front_view.jsonl`.

Pipeline QA JSONL:

```bash
sh scripts/generate_dataset/generate_internvl_dataset_pipeline.sh
```

Python entry:

```bash
python navsim/planning/script/run_generate_dataset_pipeline.py agent=recogdrive_agent train_test_split=navtrain
```

Output in the current script is `navsim_traj_base_pipeline_v5.jsonl`.

Important: `run_generate_dataset_pipeline.py` contains hard-coded local VQA
cache directories and blank OpenAI-compatible client configuration. Parameterize
or patch those before large runs. The pipeline creates QA for scene description,
traffic congestion, traffic lights, road signs, driving influence, VRU, motion
prediction, plan labels, plan explanation, driving behavior, optional trajectory
results, distance calculation, and 3D information.

## Training and Cache Commands

Environment setup:

```bash
conda env create -f environment.yml
pip install -e .
```

The current Mogo cache/training/evaluation command sequence is documented in
`Current Mogo Workflow`. Keep this section from drifting into a second source of
truth; update the workflow section when an entry script changes.

Legacy InternVL and generic NAVSIM scripts still exist in `scripts/`, but they
are not the active Mogo Qwen3 flow. Use them only when intentionally debugging
older experiments.

## Evaluation Details

Main ReCogDrive PDM scoring entry:

```bash
navsim/planning/script/run_pdm_score_recogdrive.py
```

It:

- initializes distributed NCCL,
- instantiates `recogdrive_agent`,
- loads metric cache,
- intersects scene tokens with available metric cache tokens,
- computes predicted trajectories,
- computes PDM score via `navsim.evaluate.pdm_score.pdm_score`,
- also records average XY `l2_loss`,
- writes a timestamped CSV under `${NAVSIM_EXP_ROOT}/${experiment_name}/...`.

For VQA:

- DriveLM script: `vqa_evaluation/DriveLM/infer_recogdrive_drivelm.sh`.
- LingoQA inference: `vqa_evaluation/LingoQA/infer_recogdrive.py`.
- DriveBench code lives under `vqa_evaluation/DriveBench/`.

## Validation and Test Commands

There is no broad unit-test suite, but this branch adds one targeted file:
`tests/test_qwen3_recogdrive_integration.py`. Its config/planner/select cases
run without GPU or data; the processor/model and Mogo-pkl cases need the local
Qwen3 weights and Mogo dataset on disk.

Run the no-data subset (fast, recommended after editing planner/agent/features):

```bash
python -m pytest tests/test_qwen3_recogdrive_integration.py -k "config or planner or select" -q
```

Low-cost syntax checks:

```bash
python -m py_compile navsim/agents/recogdrive/recogdrive_agent.py
python -m py_compile navsim/agents/recogdrive/recogdrive_backbone.py
python -m py_compile navsim/agents/recogdrive/recogdrive_features.py
python -m py_compile navsim/agents/recogdrive/recogdrive_diffusion_planner.py
python -m py_compile navsim/planning/script/run_training_recogdrive.py
python -m py_compile navsim/planning/script/run_dataset_caching_multi_node.py
python -m py_compile navsim/planning/script/run_pdm_score_recogdrive.py
python -m py_compile navsim/planning/script/run_training_recogdrive_rl.py
bash -n scripts/cache_dataset/run_mogo_metric_caching.sh
bash -n scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh
bash -n scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh
bash -n scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh
bash -n scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh
bash -n scripts/visualization/plt_csv_all.sh
python -m py_compile scripts/vis_scenario_video.py
```

Functional smoke checks when data/GPU are available:

```bash
# Prefer Hydra overrides on the current Mogo/Qwen3 Python entries instead of running full shell scripts.
python navsim/planning/script/run_mogo_metric_caching.py train_test_split=trainval train_test_split.scene_filter.max_scenes=2
torchrun --nproc_per_node=1 navsim/planning/script/run_training_recogdrive.py agent=recogdrive_agent train_test_split=mogo_trainval trainer.params.fast_dev_run=true
torchrun --nproc_per_node=1 navsim/planning/script/run_training_recogdrive_rl.py agent=recogdrive_agent train_test_split=mogo_trainval trainer.params.fast_dev_run=true
torchrun --nproc_per_node=1 navsim/planning/script/run_pdm_score_recogdrive.py train_test_split=mogo_trainval train_test_split.scene_filter.max_scenes=2 agent=recogdrive_agent
```

Do not run full cache/training/evaluation commands as a casual test. They can
consume large GPU hours and 1-2 TB of disk.

## Dependency Notes

NAVSIM package:

- `setup.py` installs `navsim==1.1.0` with `requirements.txt`.
- `environment.yml` declares Python 3.9, but the active conda env
  (`recogdrive`) actually runs **Python 3.10**. Treat the live interpreter as
  source of truth.
- `requirements.txt` pins `numpy==1.26.4`; keep this or newer compatible
  behavior for metric caching, because older NumPy caused metric-cache
  inconsistency.
- Core dependencies include nuPlan devkit, Hydra, PyTorch, PyTorch Lightning,
  diffusers, timm, Ray, OpenCV, pandas, Pillow, Shapely/geopandas/rasterio, and
  TensorBoard.

VLM / transformers version (critical for Qwen3):

- The active env has **`transformers==4.57.1`**, which is what exposes
  `Qwen3VLForConditionalGeneration`, `Qwen3VLProcessor`, and
  `Qwen3VLVideoProcessor`. The legacy InternVL pin (`transformers==4.37.2`
  below) predates Qwen3-VL and cannot load it. Do not "restore" 4.37.2 in the
  runtime env that runs the Qwen3 ReCogDrive scripts.
- The InternVL backbone still loads via `trust_remote_code=True`, so it works
  off the model repo's own modeling code under the newer transformers.

InternVL stack (Stage 1 SFT only):

- `internvl_chat/pyproject.toml` pins `transformers==4.37.2`,
  `tokenizers==0.15.1`, `deepspeed==0.13.5`, `timm==0.9.12`, PEFT,
  bitsandbytes, accelerate, FastAPI/Gradio utilities, and related packages.
  These pins govern the separate InternVL SFT environment, not the ReCogDrive
  planner/Qwen3 runtime.

## Editing Guidance

- For ReCogDrive model changes, start with:
  `recogdrive_agent.py`, `recogdrive_features.py`,
  `recogdrive_diffusion_planner.py`, and `recogdrive_dit.py`.
- For data/cache issues, start with:
  `navsim/common/dataclasses.py`, `navsim/common/dataloader.py`,
  `navsim/planning/training/dataset.py`, and the relevant script in
  `navsim/planning/script/`.
- For metric/PDM issues, start with:
  `navsim/planning/metric_caching/`, `navsim/evaluate/pdm_score.py`, and
  `navsim/planning/simulation/planner/pdm_planner/`.
- For VLM SFT issues, inspect `internvl_chat/shell/data_info/` and the active
  InternVL shell script before modifying upstream `internvl_chat/internvl/`.
- Preserve cache file names unless intentionally migrating caches:
  `internvl_feature.gz`, `trajectory_target.gz`, and `metric_cache.pkl`.
- When changing tensor shapes, update collate functions, feature builders,
  planner config, checkpoint filtering assumptions, and this file together.
- When adding a data adapter, write down the exact source-to-NAVSIM field mapping
  and test at least one scene through `Scene.from_scene_dict_list`,
  `Scene.get_agent_input()`, and `Scene.get_future_trajectory()`.

## Known Local State at Creation

At creation time, the worktree already had local modifications in training,
cache, evaluation, visualization, and default training config files, plus the
untracked PDF. These are treated as user/project state, not as changes to revert.
Only this `CLAUDE.md` file was intentionally added by the agent creating it.
