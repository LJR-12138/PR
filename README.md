# Fur Seal Face Intelligence Toolkit

An end-to-end research pipeline for fur-seal face **detection**, **recognition**, and **behavior informatics**. The project leverages transfer learning on modern vision backbones and geospatial analytics to transform raw expedition imagery into actionable insights on fur-seal populations around Victoria.

---

## 1. Project Overview

| Module | Goal | Key Techniques |
| --- | --- | --- |
| Detection | Localise every fur-seal face in expedition imagery | Ultralytics YOLOv8 fine-tuning, custom augmentations |
| Recognition | Tell individuals apart across photos | Face embeddings (FaceNet), metric learning, clustering |
| Behavior Informatics | Understand social/spatial patterns | GPS EXIF mining, spatiotemporal joins, network analytics |

The repository is structured to scale from quick experiments on the light-weight sample dataset to full 30 GB training corpora stored offline.

---

## 2. Getting Started

### 2.1 Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### 2.2 Fetch the datasets

1. **Clone the source repository with Git LFS enabled**
   ```powershell
   git lfs install
   git clone https://github.com/tulip-lab/furseal.git data\furseal
   ```
2. **Datasets of interest**
   - `data/furseal/datasets` – labelled YOLO-format faces for supervised detection.
   - `data/furseal/task_datasets` – large unlabelled corpora used for inference and behavior analysis.

> 💡 Store the original 30 GB archives outside of the Git repo. Symbolic links or environment variables can point to the actual storage drive.

### 2.3 Configure paths and hyper-parameters

Copy the template and edit it as needed:

```powershell
Copy-Item configs\example_config.yaml configs\local.yaml
```

Update `configs/local.yaml` to match your data layout.

### 2.4 Quick smoke test

```powershell
python -m scripts.check_setup --config configs\local.yaml
```

This validates dataset paths, GPU availability, and dependency versions.

---

## 3. Workflow

1. **Label audit / augmentation**: Optional scripts prepare custom annotations or expand the training corpus.
2. **Detection training**: Fine-tune YOLOv8 on labelled faces.
3. **Face extraction**: Batch crop detections, quality-filter, and align faces.
4. **Recognition training**: Learn embeddings via FaceNet (metric learning on positive/negative pairs).
5. **Identity clustering**: Use FAISS + DBSCAN/HDBSCAN to group embeddings into individuals.
6. **Behavior analytics**: Merge identities with GPS timestamps to infer co-occurrence patterns.

---

## 4. Command Line Recipes

```powershell
# 1. Convert raw annotations (if you have CSV/COCO etc.)
python -m scripts.prepare_detection_data --config configs\local.yaml

# 2. Fine-tune YOLOv8
python -m scripts.train_detector --config configs\local.yaml

# 3. Run detector on unlabelled archive
python -m scripts.run_detection --config configs\local.yaml --split valid

# 4. Build face embeddings
python -m scripts.build_embeddings --config configs\local.yaml

# 5. Cluster identities
python -m scripts.cluster_identities --config configs\local.yaml

# 6. Behaviour insights
python -m scripts.analyze_behaviour --config configs\local.yaml

# 7. Run the full demo pipeline (auto-clean outputs, optional retrain)
python -m scripts.run_demo --config configs\local.yaml --split valid
```

> 若需从头再训练，可附加 `--retrain`，或使用 `--limit` 控制演示时处理的原始图像数量。

Each command provides CLI arguments for overrides (see `--help`).

---

## 5. Repository Layout

```
configs/               # YAML configuration templates
scripts/               # CLI entry points that orchestrate the pipeline
src/
   analysis/            # Behaviour informatics & reporting
   data/                # Dataset utilities & download helpers
   detection/           # YOLO pipeline wrappers
   recognition/         # Embedding models & matching logic
   utils/               # Generic helpers (logging, paths, io)
tests/                 # Unit/functional tests
```

> ℹ️ Output folders under `outputs/` stay empty until you run the pipeline; each script recreates them on demand so artefacts never mix between experiments.

---

## 6. Training Guidance

- **Hardware**: A CUDA-capable GPU (≥8 GB VRAM) accelerates both YOLO fine-tuning and FaceNet embedding extraction. CPU-only mode works for prototyping but is slow.
- **Run naming**: The default detector experiment now writes to `outputs/training/detector/exp_gpu_augmix`; adjust `configs/local.yaml` if you need parallel runs.
- **Mixed precision**: Enabled by default on GPU to fit larger batches.
- **Checkpointing**: All training scripts log to `outputs/training/<run_name>` with tensorboard-ready metrics.
- **Evaluation**: mAP (detection) and clustering purity/NMI (recognition) reported in logs.

---

## 7. Behaviour Analytics Ideas

- Construct co-occurrence graphs (nodes=individual fur seals, edges=joint sightings) and apply community detection.
- Map visit frequency per beach using GPS coordinates (geohash clustering).
- Study temporal routines by binning sightings into tidal/diurnal segments.

---

## 8. Contributing & Next Steps

- Use `pre-commit install` to enable linting/formatting hooks.
- Open issues for dataset-specific challenges (labelling noise, occlusions, lighting).
- Extend to multi-modal inputs (acoustic tags, drone footage) where available.

---

## 9. References

- Ultralytics YOLOv8: https://docs.ultralytics.com/
- FaceNet PyTorch: https://github.com/timesler/facenet-pytorch
- DeepFace: https://github.com/serengil/deepface
- DogFaceNet: https://github.com/marvinTeichmann/dogFaceNet

Happy seal spotting! 🦭

---

## 10. 演示与复现（7–8 分钟）

- 演示脚本（逐分钟旁白与命令）：参见 `docs/DEMO_SCRIPT.md`
- 项目报告（方法/实验/结果/改进）：参见 `docs/REPORT.md`

快速运行最小命令集（Windows/PowerShell）：

```powershell
# 假设已激活虚拟环境：& .\.venv\Scripts\Activate.ps1

# 1) 检测
.\.venv\Scripts\python.exe -m scripts.run_detection --split valid

# 2) 构建特征
.\.venv\Scripts\python.exe -m scripts.build_embeddings

# 3) 聚类身份
.\.venv\Scripts\python.exe -m scripts.cluster_identities

# 4) 行为分析
.\.venv\Scripts\python.exe -m scripts.analyze_behaviour
```

如需可重复配置与更完整流程，请参考上文 “Command Line Recipes” 中带 `--config configs\local.yaml` 的命令或直接运行 `scripts.run_demo`。
