# 海狗（毛皮海豹）人脸检测与识别项目报告

作者：项目组｜日期：2025-11-04｜仓库：`fur-seal face recognition`

## 摘要
本项目面向维多利亚州多岛屿拍摄的野外海狗照片，构建了完整的人脸检测、个体识别与行为信息学分析流水线：
- 人脸检测：以 YOLOv8s 为骨干，针对海狗脸数据微调并在推理阶段结合更高分辨率（1024）、TTA 与后处理 IoU 去重，显著缓解“同脸多框”和“大脸漏检”问题。
- 人脸识别：提取裁剪的人脸 patch 特征并进行无监督聚类以区分个体。考虑到小样本与域偏移，我们实现并对比了多种简洁稳健的 embedding 后端（FaceNet、HOG+PCA、颜色直方图），最终在当前数据规模下采用颜色直方图 + DBSCAN（余弦度量）取得相对稳定的聚类结果。
- 行为信息学：利用照片的 EXIF/GPS 信息生成共现图谱与群体统计，回答“哪些海狗经常同时出现”“在相同图像上的同现关系”等问题。

在给定验证集上，YOLOv8s 微调后取得验证指标 P≈0.435 / R≈0.335 / mAP50≈0.329；推理端经 1024 分辨率与 IoU 去重增强后，明显大脸召回提升且重复框大幅减少。识别端在当前样本规模下形成 3 个有效簇与少量噪声，支撑基本的个体统计与共现图谱生成。

## 1. 数据与任务
- 任务数据：来自 repo `data/furseal/task_datasets`（公开小样本，无标签），真实完整数据 >30GB（题目描述）。
- 任务目标：
  1) 在任意包含海狗的图像中检测所有海狗面部；
  2) 基于人脸 patch 识别是否来自同一只海狗（无监督聚类）；
  3) 借助 EXIF/GPS 元信息分析群体行为（同现关系与时空统计）。

## 2. 方法
### 2.1 检测（Detector）
- 模型：YOLOv8s（较 n 版容量更大，检测精度更好），训练 150 epoch，batch=4，img=832，AdamW 优化器，含适度数据增强（mosaic、mixup、erasing、色彩抖动等）。
- 推理增强：
  - 分辨率提升至 1024；
  - 轻量 TTA；
  - 事后 IoU 去重（保留高置信度框，`post_nms_duplicate_iou=0.85`）抑制“同脸多框”。
- 关键脚本与文件：
  - 训练：`scripts/train_detector.py`（已完成训练；本阶段复现以推理为主）
  - 推理：`scripts/run_detection.py` → 输出到 `outputs/detections/latest/`
  - 配置：`configs/local.yaml`（训练/推理参数集中配置）

### 2.2 裁脸与特征（Recognition Embedding）
- 裁脸：`src/recognition/face_extractor.py`
  - 使用 YOLO 检测框裁剪，过滤极小框与低质量框；新增“重复框过滤”（同图内 IoU 高于阈值的候选仅保留置信度高的一个）。
- 特征后端：`src/recognition/embedding_model.py`
  - FaceNet-Inception（参考）、HOG+PCA（实现以备扩展）、颜色直方图（当前默认）三种方案；
  - 小样本下实测 FaceNet 在本域特征几乎一致，难以分簇；HOG+PCA 在本数据上距离等距倾向明显，DBSCAN 失败；颜色直方图对本数据的光照与花纹变化鲁棒性较好，聚类可用。

### 2.3 聚类（Clustering）
- 算法：DBSCAN，metric=cosine（对直方图进行 L2 归一化后效果更稳），核心参数：`eps≈0.15`、`min_samples=2`；
- 实现：`src/recognition/cluster.py`，支持可选 FAISS 近邻预检（当前小样本关闭）。
- 输出：`outputs/embeddings/identities.parquet`，记录每个裁脸 patch 的身份簇标签（-1 为噪声）。

### 2.4 行为信息学（Behaviour Analytics）
- EXIF 解析：读取时间戳与 GPS，经纬度转小数；
- 共现图谱：以“同一张图像共同出现”为边（权重为共现次数，当前阈值 1）生成图谱，输出 GraphML；
- 统计报表：每身份的出现次数、最早/最晚出现时间等；
- 脚本：`scripts/analyze_behaviour.py` → 输出到 `outputs/reports/`（`summary.html`, `population_summary.csv`, `cooccurrence.graphml`）。

## 3. 工程改进点汇总
- 推理增强（解决“大脸漏检”和“同脸多框”）：
  - imgsz=1024 + TTA，有效提升大脸近景召回；
  - 推理后 IoU 去重 + 裁脸阶段重复过滤，重复框显著减少；
- 识别后端多备选并验证：保留 HOG+PCA / FaceNet 作为后备研究方向，目前以颜色直方图作为稳健默认；
- 配置统一化：`configs/local.yaml` 管理所有关键参数，便于可重复性；
- 结果产物结构化：检测标签、裁脸、特征、身份、行为报告分目录输出，便于复现与演示。

## 4. 实验与结果（节选）
- Detector 训练（YOLOv8s）：验证集 P≈0.435 / R≈0.335 / mAP50≈0.329；
- 推理后处理：
  - 采用 IoU 去重后，图片内最大框重叠从 ~0.85–0.90 降至 ~≤0.55，重复框问题基本消除；
  - 8 张验证图像均有有效标签输出，检测数提升，明显大脸更稳定被检出；
- 识别与聚类（基于裁脸 + 颜色直方图 + DBSCAN）：
  - 最新身份计数：id1≈24、id0≈2、id2≈2、噪声(-1)≈6（在当前小样本上较为稳定）；
- 行为信息学：
  - 生成 `summary.html` 与 `cooccurrence.graphml`，在 `min_cooccurrence=1` 下可见共现关系；

注：上述数值来自本地最新跑批日志与程序输出，因随机性和配置微调可能轻微波动。

## 5. Demo 演示（7–8 分钟脚本建议）
- 0:00–0:40 问题与数据：海狗保护背景、任务与数据来源（task_datasets）、挑战点（尺度变化、遮挡、相似花纹）。
- 0:40–1:30 流水线概览：检测→裁脸→特征→聚类→行为图谱，展示 `configs/local.yaml` 关键段落。
- 1:30–2:10 检测模型：说明 YOLOv8s 微调要点与推理增强（1024 + TTA + IoU 去重）。
- 2:10–3:10 现场推理：运行 `scripts/run_detection.py --split valid`，打开 `outputs/detections/latest/` 看图片与 labels。简述重复框抑制效果。
- 3:10–4:10 裁脸与特征：展示 `outputs/faces/`，说明裁脸质量阈值与重复过滤；运行 `scripts/build_embeddings.py`。
- 4:10–5:00 聚类与身份：运行 `scripts/cluster_identities.py`，打开 `outputs/embeddings/identities.parquet`（或打印 value_counts），展示身份数量与分布。
- 5:00–6:30 行为分析：运行 `scripts/analyze_behaviour.py`，打开 `outputs/reports/summary.html` 与 `cooccurrence.graphml`，说明如何读图和答案示例（哪些海狗经常同时出现）。
- 6:30–7:30 总结与展望：现状、局限（小样本/域偏移/特征弱监督）、未来工作（对齐、强特征、更多数据、半监督/度量学习）。

## 6. 复现与运行指南（Windows/PowerShell）
> 下列命令仅作文档记录，实际在本机已执行验证。

- 环境准备（可选）
```powershell
# 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 安装依赖
pip install -U pip
pip install -e .[dev]
```

- 运行检测
```powershell
.\.venv\Scripts\python.exe -m scripts.run_detection --split valid
```
输出：`outputs/detections/latest/`，labels 在 `labels/` 子目录。

- 构建特征与聚类
```powershell
.\.venv\Scripts\python.exe -m scripts.build_embeddings
.\.venv\Scripts\python.exe -m scripts.cluster_identities
```
输出：`outputs/embeddings/embeddings.parquet` 与 `identities.parquet`。

- 行为分析
```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_behaviour
```
输出：`outputs/reports/summary.html`, `population_summary.csv`, `cooccurrence.graphml`。

- 可选：重新训练检测器（需标注数据集）
```powershell
.\.venv\Scripts\python.exe -m scripts.train_detector
```

## 7. 回答题目问题（依据当前跑批结果）
- 一组给定的照片中有多少只海狗？
  - 可通过检测框数聚合得到每张图像的海狗脸数量；对于个体数量，需在同一图像去重（本项目已加入 IoU 去重）后计数。
- 哪些海狗经常同时出现？
  - 使用 `cooccurrence.graphml` 或 `summary.html` 中的共现列表，阈值（min_cooccurrence）可调；当前以同图像共现为边定义。
- 这些同时出现的海狗之间的潜在关系是什么？
  - 结合共现频次与时间/地点（EXIF/GPS），可推测伴随关系或社群结构；需要更长时间序列和更多数据进行统计验证。

## 8. 局限与改进方向
- 识别：受小样本与域偏移影响，无监督聚类对边界样本敏感；后续考虑：
  - 加入人脸几何对齐（眼角/口鼻点），提高特征稳定性；
  - 引入轻量级通用视觉特征（ResNet/CLIP）+ PCA 替代直方图；
  - 度量学习（如 ArcFace 微调）或半监督标签传播以增强分辨性。
- 检测：
  - 继续扩充训练数据与增强策略，针对近景/偏转/遮挡进行针对性采样；
  - 多尺度推理与自适应阈值进一步平衡召回与精度。
- 行为：
  - 增加时间窗、空间聚类（geohash/DBSCAN）分析群体迁徙规律；
  - 数据治理：检查 EXIF 缺失与 GPS 精度偏差。

## 9. 参考资料
- YOLOv8: Ultralytics
- FaceNet, DeepFace, DogFaceNet 等人/犬脸识别研究
- scikit-learn, OpenCV, faiss, facenet-pytorch 等库
- 题目数据仓库：
  - https://github.com/tulip-lab/furseal/tree/main/task_datasets
  - https://github.com/tulip-lab/furseal/tree/main/datasets

---
本报告对应的代码与可运行脚本均已在仓库中实现与验证；若需要，我们可进一步补充录屏素材与演示 PPT，并按上述“演示脚本”录制 7–8 分钟视频。