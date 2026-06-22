# MERTech 实验管理指南

本文档说明三层分离（**配置 / Run / Registry**）的目录结构、日常用法，以及从旧脚本迁移的方式。

---

## 架构概览

| 层级 | 目录 | 职责 |
|------|------|------|
| **配置层** | `configs/` | 实验超参 YAML，不改代码即可切换实验 |
| **Run 层** | `runs/guzheng/` | 每次训练的自包含目录（快照、checkpoint、指标、测试结果） |
| **Registry 层** | `experiments/registry.yaml` | 实验总表，记录 run 路径、checkpoint、备注与关键结果 |

核心代码仍在 `function/`（`model.py`、`fit.py`、`metrics_ipt.py` 等），由 `function/experiment_config.py` 把 YAML 写入 `function/config.py` 的运行时变量。

```
configs/*.yaml
    ↓  scripts/train.py
runs/guzheng/{date}_{experiment_id}/
    ├── config.yaml          # 训练时复制的配置快照
    ├── run_meta.json        # git commit、命令、best checkpoint
    ├── best_e_{epoch}       # 当前仅保留最优 checkpoint
    ├── ipt_metrics.jsonl    # 每轮验证指标
    ├── failure_inspection/  # 可选失败样本图
    └── test/
        ├── latest.json      # 最近一次测试结果
        └── *.txt / *.json   # 带时间戳的测试报告

experiments/registry.yaml    # 人工维护的实验索引
```

---

## 快速开始

### 训练

```bash
# Baseline（无 FPT，checkpoint 按 IPT frame F1）
python scripts/train.py --config configs/baseline.yaml

# FPT + combined checkpoint（IPT + pitch + PN frame + PN event）/ 4
python scripts/train.py --config configs/fpt_combined_pn.yaml

# 指定输出目录（不自动建 runs/guzheng/...）
python scripts/train.py --config configs/baseline.yaml --run-dir runs/guzheng/my_manual_run
```

训练结束后终端会打印 `run_dir` 和 `best checkpoint` 路径，以及测试命令。

### 测试

```bash
# 推荐：从 run 目录读 config 快照，自动找 best_e_*
python scripts/test.py --run runs/guzheng/2026-06-18_fpt_combined_pn

# 指定 checkpoint
python scripts/test.py --run runs/guzheng/2026-06-18_fpt_combined_pn \
  --checkpoint runs/guzheng/2026-06-18_fpt_combined_pn/best_e_2555

# 旧模型（无 run 快照）：用 config + checkpoint
python scripts/test.py \
  --config configs/baseline.yaml \
  --checkpoint data/model/baseline/best_e_2225

# 自定义测试集 split（默认 test）
python scripts/test.py --run runs/guzheng/... --test-group test,validation
```

测试结果默认写入 `<run_dir>/test/`，并更新 `test/latest.json`。无 `--run` 时仍写入 `output/test_results/`。

### 查看帮助

```bash
python scripts/train.py --help
python scripts/test.py --help
```

---

## 配置文件说明

配置文件为嵌套 YAML，只需写与默认值不同的字段。完整默认值由 `function/experiment_config.py` 的 `default_experiment_dict()` 从 `function/config.py` 生成。

### 示例：`configs/baseline.yaml`

```yaml
experiment_id: baseline
seed: 42
dataset: Guzheng_Tech99

model:
  url: m-a-p/MERT-v1-95M
  freeze_all: false
  use_fpt: false

training:
  best_checkpoint_metric: ipt   # ipt | pitch | combined | pn_frame
  early_stopping: 1000
  validation_interval: 5

loss:
  pitch_weight: 0.5
  onset_weight: 1.0
```

### 示例：`configs/fpt_combined_pn.yaml`

```yaml
experiment_id: fpt_combined_pn

model:
  use_fpt: true
  fpt_levels: 3
  fpt_num_layers: 1
  fpt_num_heads: 8
  fpt_dropout: 0.1

training:
  best_checkpoint_metric: combined
```

### `best_checkpoint_metric` 选项

| 值 | 公式 | 说明 |
|----|------|------|
| `ipt` | IPT micro frame F1 | baseline / 整体 frame 最优 |
| `pitch` | pitch frame F1 | 盯 pitch 辅助任务 |
| `combined` | (IPT + pitch + PN frame + PN event) / 4 | PN frame 与 event 都纳入选模 |
| `pn_frame` | **(IPT + pitch + 2 × PN frame) / 4** | 侧重 PN frame，PN 项权重 ×2，比纯 PN 更稳 |

- IPT / pitch 来自 legacy 验证（固定 0.5 阈值）
- PN frame / event（`combined` 用）来自 per-class threshold sweep（若开启）

**`pn_frame` 配置示例**：`configs/fpt_pn_frame.yaml`

```yaml
training:
  best_checkpoint_metric: pn_frame
```

训练日志与 `ipt_metrics.jsonl` 的 `extra` 字段会记录分解项，例如：

```json
{
  "checkpoint_metric": "pn_frame",
  "checkpoint_score": 0.72,
  "ipt_frame_f1": 0.89,
  "pitch_frame_f1": 0.91,
  "pn_frame_f1": 0.42,
  "pn_frame_weight": 2
}
```

终端示例：

```
best_ckpt_metric (pn_frame): 0.7200  [IPT frame=0.8900  pitch=0.9100  PN frame=0.4200  (PN weight=2)]
```

### 主要配置字段映射

| YAML 路径 | `config.py` 变量 |
|-----------|------------------|
| `experiment_id` | `EXPERIMENT_ID`, `saveName` 前缀 |
| `model.url` | `URL`, `MERT_SAMPLE_RATE` |
| `model.use_fpt` | `USE_FPT` |
| `model.fpt_*` | `FPT_LEVELS`, `FPT_NUM_LAYERS`, … |
| `training.best_checkpoint_metric` | `BEST_CHECKPOINT_METRIC` |
| `training.early_stopping` | `EARLY_STOPPING` |
| `loss.pitch_weight` | `PITCH_LOSS_WEIGHT` |
| `loss.onset_weight` | `ONSET_LOSS_WEIGHT` |
| `eval.*` | `EVAL_*`, `THRESHOLD_*`, `FAILURE_*` |
| `runtime.cuda_device` | `CUDA_VISIBLE_DEVICES` |

---

## Registry 用法

`experiments/registry.yaml` 用于记录「做了哪些实验、产物在哪、结果如何」，便于写论文和做消融表。

```yaml
experiments:
  - id: baseline
    status: done          # planned | running | done
    config: configs/baseline.yaml
    run_dir: runs/guzheng/2026-06-18_baseline
    checkpoint: runs/guzheng/2026-06-18_baseline/best_e_2225
    notes: "Test PN event F1 ~51%."
```

每次完成 train + test 后，手动更新对应条目（后续可加 `scripts/summarize_runs.py` 自动汇总）。

---

## 消融实验建议

**一次只改一个变量**，其余锁死在 baseline config 上：

| ID | 改动 | config 文件 |
|----|------|-------------|
| A0 | baseline | `configs/baseline.yaml` |
| A1 | + FPT | `use_fpt: true`, `best_checkpoint_metric: ipt` |
| A2 | + FPT + combined ckpt | `configs/fpt_combined_pn.yaml` |
| A3 | + FPT + pn_frame ckpt | `configs/fpt_pn_frame.yaml` |
| A4 | + focal loss（待实现） | 复制 A3，改 `loss.ipt_loss` |

流程：

1. 复制 `configs/baseline.yaml` → `configs/ablations/A1_fpt.yaml`
2. `python scripts/train.py --config configs/ablations/A1_fpt.yaml`
3. `python scripts/test.py --run runs/guzheng/<...>`
4. 更新 `experiments/registry.yaml`

---

## 与旧脚本的对应关系

| 旧方式 | 新方式 |
|--------|--------|
| 改 `function/config.py` 再 `python run.py` | `python scripts/train.py --config configs/....yaml` |
| 改 `test_train_frame_ao` 里的 `CHECKPOINT_PATH` | `python scripts/test.py --run ...` 或 `--checkpoint ...` |
| `data/model/{timestamp}/...` | `runs/guzheng/{date}_{experiment_id}/` |
| 手写 `output_records` | `experiments/registry.yaml` + `run_meta.json` |

`run.py` 和 `test_train_frame_ao` 仍可用，但会打印 deprecation 提示并转发到新脚本：

```bash
python run.py                                    # 默认 configs/fpt_combined_pn.yaml
python test_train_frame_ao --run runs/guzheng/...  # 需传参，见 scripts/test.py --help
```

---

## 代码改动摘要

### 新增文件

| 文件 | 说明 |
|------|------|
| `function/experiment_config.py` | 加载 YAML、合并默认值、写入 config、创建 run 目录、写快照 |
| `configs/baseline.yaml` | Baseline 实验配置 |
| `configs/fpt_combined_pn.yaml` | FPT + combined checkpoint |
| `configs/fpt_pn_frame.yaml` | FPT + PN-frame-weighted checkpoint |
| `experiments/registry.yaml` | 实验登记册 |
| `experiments/WORKFLOW.md` | 本文档 |
| `scripts/train.py` | 统一训练入口 |
| `scripts/test.py` | 统一测试入口（由原 `test_train_frame_ao` 升级） |
| `runs/guzheng/.gitkeep` | 占位目录 |

### 修改文件

| 文件 | 说明 |
|------|------|
| `function/config.py` | 新增 `EXPERIMENT_ID` / `RUN_DIR` / `SEED`；去掉 `USE_FPT` 自动绑定 `BEST_CHECKPOINT_METRIC` |
| `function/fit.py` | combined checkpoint = (IPT + pitch + PN frame + PN event) / 4 |
| `run.py` | 薄封装 → `scripts/train.py` |
| `test_train_frame_ao` | 薄封装 → `scripts/test.py` |
| `.gitignore` | 忽略 `runs/guzheng/*/`（权重不进 git） |

### Run 目录内产物

| 文件 | 内容 |
|------|------|
| `config.yaml` | 训练开始时复制的配置快照 |
| `run_meta.json` | `experiment_id`, `git_commit`, `command`, `best_checkpoint`, 起止时间 |
| `best_e_{epoch}` | 当前最优模型权重 |
| `ipt_metrics.jsonl` | 每轮验证完整指标（含 per-class、checkpoint_score） |
| `test/latest.json` | 最近一次 test 的完整 JSON 结果 |

---

## 已知限制与后续可做

1. **仅保留一个 `best_e_*`**：其他 epoch 权重会被覆盖；若需 retrospective 选模，需在 `fit.py` 增加 `epoch_e_{n}` 定期保存。
2. **Registry 需手动更新**：训练脚本不会自动写 `registry.yaml`。
3. **Focal loss / 多 config 批量跑**：尚未实现，可在 `configs/ablations/` 下扩展。
4. **建议后续**：`scripts/summarize_runs.py` 扫描 `runs/guzheng/*/test/latest.json` 生成 PN 对比 CSV。

---

## 环境

在 `guzheng` conda 环境中运行（需 `PyYAML`、`torch`、`transformers` 等，见 `requirements.txt`）：

```bash
conda activate guzheng
cd /path/to/MERTech
python scripts/train.py --config configs/baseline.yaml
```
