# EasyMetrics

一个轻量级、零依赖的机器学习指标评估平台，专注于提供简单易用且准确的模型评估工具。

## ✨ 核心特性

### 🔧 设计理念
- **零依赖**: 仅依赖 Python 和 numpy，无需安装大型深度学习框架
- **模块化架构**: 清晰的代码结构，易于扩展和定制
- **用户友好**: 简洁的 API 设计，一行代码完成评估
- **性能优化**: 支持并行计算，大幅提升评估速度

### 📊 功能特性

#### 目标检测评估
- **完整的 COCO 指标支持**:
  - mAP (IoU 0.5:0.95) - 综合评估指标
  - mAP_50 (IoU 0.5) - 宽松条件下的性能
  - mAP_75 (IoU 0.75) - 严格条件下的性能
  - mAP_s/m/l - 不同尺度目标的性能
  - AR_1, AR_10, AR_100 - 不同最大检测数下的召回率
- **每类别独立评估**: 详细的类别级指标
- **独家功能**: 自动计算满足特定精度要求的最佳置信度阈值

#### 分类评估
- **F1 Score 指标支持**:
  - 二分类和多分类场景
  - 宏平均、微平均和加权平均
  - 同时计算精确率、召回率和F1 Score
- **AUC 指标支持**:
  - 二分类场景的ROC曲线下面积
  - 线性插值和梯形积分两种计算方法
  - 衡量模型对正负样本的区分能力

#### 数据格式支持
- **多种输入格式**: 支持 COCO、VOC、YOLO 等常见格式
- **自动类型转换**: 支持标量、列表、numpy 数组等多种输入类型
- **灵活的格式指定**: 可分别指定预测值和真值的格式
- **智能格式检测**: 自动识别常见的数据格式

#### 用户体验
- **进度条功能**: 直观展示评估进度，可选择禁用
- **并行计算**: 支持多核并行加速，可指定线程数
- **详细的指标解释**: 提供清晰的指标含义说明
- **丰富的示例代码**: 覆盖各种使用场景
- **错误处理**: 完善的边界情况处理和错误提示

### 🎯 技术优势
- **高精度计算**: 采用 COCO 标准的 101 点插值法，计算更准确
- **多尺度评估**: 支持对不同大小目标的独立评估
- **内存优化**: 高效的内存管理，支持大规模数据集
- **扩展性强**: 模块化设计，易于添加新的评估任务

## 📋 参数详解

### evaluate_detection 函数参数说明

#### 核心参数

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **preds** | List[Any] | 必填 | 每张图片的预测结果列表。支持多种格式，具体取决于 `format` 或 `pred_format` 参数。 |
| **targets** | List[Any] | 必填 | 每张图片的真实标签列表。支持多种格式，具体取决于 `format` 或 `target_format` 参数。 |
| **metrics** | Optional[List[str]] | None | 需要返回的特定指标列表。如果为 None，则返回所有计算的指标。 |
| **n_jobs** | int | 1 | 并行计算线程数。1 表示串行计算，-1 表示使用所有可用 CPU 核心。 |
| **score_criteria** | Optional[List[Tuple[float, float]]] | None | 计算指定 IoU 和精度下的最佳置信度阈值。格式为 `[(iou_thresh, min_precision), ...]`，例如 `[(0.5, 0.9)]` 表示寻找 IoU=0.5 时精度至少为 0.9 的最低置信度。 |
| **progress** | bool | True | 是否显示进度条。默认为 True，在计算过程中显示评估进度。 |

#### 格式参数

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **format** | str | "coco" | 输入数据的默认格式，当 `pred_format` 和 `target_format` 未指定时使用。支持 "coco", "voc", "yolo", "custom"。 |
| **pred_format** | Optional[str] | None | 预测结果的格式，优先级高于 `format`。 |
| **target_format** | Optional[str] | None | 真实标签的格式，优先级高于 `format`。 |

#### 额外参数（**kwargs）

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **image_size** | Tuple[int, int] | (640, 640) | YOLO 格式需要的图像尺寸 (width, height)，用于将归一化坐标转换为绝对坐标。 |
| **custom_converter** | Callable | None | 自定义格式的转换函数，当 `format` 为 "custom" 时使用。 |

## 📦 安装方法

### 从 PyPI 安装 (推荐)

```bash
pip install EasyMetrics
```

### 从 GitHub 安装

```bash
pip install git+https://github.com/Lucidyn/EasyMetrics.git
```

### 依赖项
- **核心依赖**: numpy >= 1.17.0
- **可选依赖**: tqdm (用于进度条)

## 🚀 快速上手

### 核心接口用法

EasyMetrics 提供了三个核心评估接口，满足不同场景的需求：

#### 1. 统一评测接口

`evaluate` 函数是统一的评测入口，支持自动检测任务类型，一行代码完成评估：

```python
from easyMetrics import evaluate

# 分类任务（自动检测）
class_result = evaluate(class_preds, class_targets)
print(f"F1 Score: {class_result['f1']:.4f}")
print(f"AUC: {class_result['auc']:.4f}")

# 检测任务（自动检测）
det_result = evaluate(det_preds, det_targets)
print(f"mAP: {det_result['mAP']:.4f}")
print(f"mAP_50: {det_result['mAP_50']:.4f}")
```

#### 2. 检测任务专用接口

`evaluate_detection` 函数专门用于目标检测评估，提供完整的 COCO 指标：

```python
from easyMetrics import evaluate_detection

# 目标检测评估
det_result = evaluate_detection(det_preds, det_targets)
print(f"mAP: {det_result['mAP']:.4f}")
print(f"mAP_50: {det_result['mAP_50']:.4f}")
print(f"mAP_75: {det_result['mAP_75']:.4f}")
```

#### 3. 分类任务专用接口

`evaluate_classification` 函数专门用于分类评估，支持 F1 Score 和 AUC 指标：

```python
from easyMetrics import evaluate_classification

# 分类评估
class_result = evaluate_classification(class_preds, class_targets)
print(f"F1 Score: {class_result['f1']:.4f}")
print(f"Precision: {class_result['precision']:.4f}")
print(f"Recall: {class_result['recall']:.4f}")
print(f"AUC: {class_result['auc']:.4f}")
```

### 基本用法

一行代码完成目标检测评估：

```python
import numpy as np
from easyMetrics import evaluate_detection

# 准备数据 - 每张图片一个字典
preds = [{
    'boxes': np.array([[10, 10, 50, 50]]),  # [x1, y1, x2, y2] 格式
    'scores': np.array([0.95]),              # 置信度分数
    'labels': np.array([0])                 # 类别索引
}]

targets = [{
    'boxes': np.array([[10, 10, 50, 50]]),  # 真实边界框
    'labels': np.array([0])                 # 真实类别
}]

# 执行评估
results = evaluate_detection(preds, targets)

# 查看结果
print(f"mAP (IoU 0.5:0.95): {results['mAP']:.4f}")
print(f"mAP_50 (IoU 0.5): {results['mAP_50']:.4f}")
print(f"mAP_75 (IoU 0.75): {results['mAP_75']:.4f}")
print(f"AR_100 (MaxDets=100): {results['AR_100']:.4f}")
```

### 完整示例

下面是一个更完整的示例，包含多个图片的评估：

```python
import numpy as np
from easyMetrics import evaluate_detection

# 准备数据 - 假设有 2 张图片

# 图片 1: 预测完全正确
preds_1 = {
    'boxes': np.array([[10, 10, 50, 50]]),
    'scores': np.array([0.95]),
    'labels': np.array([0])
}
targets_1 = {
    'boxes': np.array([[10, 10, 50, 50]]),
    'labels': np.array([0])
}

# 图片 2: 有一个误检 (FP) 和一个漏检 (FN)
preds_2 = {
    'boxes': np.array([[100, 100, 150, 150], [0, 0, 20, 20]]),
    'scores': np.array([0.9, 0.6]),
    'labels': np.array([0, 0])
}
targets_2 = {
    'boxes': np.array([[100, 100, 150, 150], [200, 200, 250, 250]]),
    'labels': np.array([0, 0])
}

preds = [preds_1, preds_2]
targets = [targets_1, targets_2]

# 执行评估
results = evaluate_detection(preds, targets)

# 查看结果
print(f"mAP (IoU 0.5:0.95): {results['mAP']:.4f}")
print(f"mAP_50 (IoU 0.5): {results['mAP_50']:.4f}")
print(f"mAP_75 (IoU 0.75): {results['mAP_75']:.4f}")
print(f"AR_100 (MaxDets=100): {results['AR_100']:.4f}")
```

### 并行计算

对于大规模数据集，启用并行计算加速：

```python
# 使用 4 个核心
results = evaluate_detection(preds, targets, n_jobs=4)

# 使用所有可用核心
results = evaluate_detection(preds, targets, n_jobs=-1)
```

### 寻找最佳阈值

自动计算满足特定精度要求的最佳置信度阈值：

```python
# 场景: IoU=0.5 时精度至少达到 90%
results = evaluate_detection(
    preds, targets,
    score_criteria=[(0.5, 0.9)]  # 格式: [(iou阈值, 最低精度要求)]
)

# 获取类别 0 的最佳阈值
best_thresh = results.get('BestScore_IoU0.50_P0.90_0')
print(f"类别 0 的推荐阈值: {best_thresh}")

# 场景: 多个精度要求
results = evaluate_detection(
    preds, targets,
    score_criteria=[(0.5, 0.9), (0.75, 0.8)]
)

# 获取不同要求下的阈值
best_thresh_50 = results.get('BestScore_IoU0.50_P0.90_0')
best_thresh_75 = results.get('BestScore_IoU0.75_P0.80_0')
print(f"IoU=0.5, P>=0.9 阈值: {best_thresh_50}")
print(f"IoU=0.75, P>=0.8 阈值: {best_thresh_75}")
```

### 自定义指标筛选

只计算你关心的指标，减少输出干扰，提高计算效率：

```python
# 只计算 mAP 和 mAP_50
results = evaluate_detection(preds, targets, metrics=['mAP', 'mAP_50'])
print(results)

# 只计算召回率相关指标
results = evaluate_detection(preds, targets, metrics=['AR_100'])
print(results)
```

### 多种数据格式

支持多种输入格式，包括 COCO、VOC、YOLO 等：

```python
# VOC 格式示例
preds_voc = [[10, 10, 50, 50, 0, 0.95]]  # [x1, y1, x2, y2, class_id, confidence]
targets_voc = [[10, 10, 50, 50, 0]]  # [x1, y1, x2, y2, class_id]

results = evaluate_detection(
    [preds_voc], [targets_voc],
    pred_format="voc",
    target_format="voc"
)
print(f"VOC 格式评估结果: {results['mAP']:.4f}")

# YOLO 格式示例
# YOLO 格式: [class_id, x_center, y_center, width, height, confidence]
# 注意: 坐标是归一化的 (0-1)
preds_yolo = [[0, 0.5, 0.5, 0.2, 0.2, 0.95]]
targets_yolo = [[0, 0.5, 0.5, 0.2, 0.2]]

results = evaluate_detection(
    [preds_yolo], [targets_yolo],
    pred_format="yolo",
    target_format="yolo",
    image_size=(640, 640)  # YOLO 格式需要的图像尺寸
)
print(f"YOLO 格式评估结果: {results['mAP']:.4f}")

# 混合格式示例
# 预测值使用 YOLO 格式，真值使用 VOC 格式
results = evaluate_detection(
    [preds_yolo], [targets_voc],
    pred_format="yolo",
    target_format="voc",
    image_size=(640, 640)
)
print(f"混合格式评估结果: {results['mAP']:.4f}")
```

### 灵活的输入格式

支持标量、列表等多种输入类型：

```python
# 标量输入示例 (单目标情况)
preds_scalar = {'boxes': [10, 10, 50, 50], 'scores': 0.95, 'labels': 0}
targets_scalar = {'boxes': [10, 10, 50, 50], 'labels': [0]}

results = evaluate_detection([preds_scalar], [targets_scalar])
print(f"标量输入评估结果: {results['mAP']:.4f}")

# 列表输入示例
preds_list = {
    'boxes': [[10, 10, 50, 50], [100, 100, 150, 150]],
    'scores': [0.95, 0.9],
    'labels': [0, 0]
}
targets_list = {
    'boxes': [[10, 10, 50, 50], [100, 100, 150, 150]],
    'labels': [0, 0]
}

results = evaluate_detection([preds_list], [targets_list])
print(f"列表输入评估结果: {results['mAP']:.4f}")
```

### 进度条控制

控制是否显示评估进度条：

```python
# 启用进度条 (默认)
results = evaluate_detection(preds, targets, progress=True)

# 禁用进度条
results = evaluate_detection(preds, targets, progress=False)
```

### 多类别评估

评估包含多个类别的检测结果：

```python
# 多类别示例
preds_multi = [{
    'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
    'scores': np.array([0.95, 0.9]),
    'labels': np.array([0, 1])  # 两个不同的类别
}]
targets_multi = [{
    'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
    'labels': np.array([0, 1])  # 两个不同的类别
}]

results = evaluate_detection(preds_multi, targets_multi)
print(f"多类别评估 mAP: {results['mAP']:.4f}")
print(f"类别 0 的 AP: {results['AP_0']:.4f}")
print(f"类别 1 的 AP: {results['AP_1']:.4f}")
```

## 📋 目录结构

```
easyMetrics/
├── easyMetrics/         # 核心代码
│   ├── core/             # 抽象基类
│   │   └── base.py
│   └── tasks/            # 任务实现
│       └── detection/    # 目标检测
│           ├── interface.py        # 对外接口
│           ├── map.py              # mAP 核心逻辑
│           ├── matcher.py          # 匹配策略
│           ├── utils.py            # 辅助函数
│           └── format_converter.py # 格式转换器
├── docs/                 # 文档
│   ├── 使用指南.md        # 详细使用指南
│   └── 指标详解.md        # 指标计算原理
├── demo.py               # 完整使用示例
├── README.md             # 项目概述
├── pyproject.toml        # 项目配置
└── LICENSE               # 许可证
```

## 📚 详细文档

- **使用指南.md**: 详细的安装、配置和使用说明，包含各种功能的完整示例
- **指标详解.md**: 详细介绍各种评估指标的计算原理和含义
- **demo.py**: 完整的代码示例，覆盖所有主要功能

## � 指标详解

### 平均精度 (Average Precision, AP)

AP 是目标检测中最重要的指标，它综合了**精度 (Precision)** 和 **召回率 (Recall)**。

- **Precision (查准率)**: 预测为正样本中，真正正确的比例。 $P = \frac{TP}{TP + FP}$
- **Recall (查全率)**: 所有真实正样本中，被正确预测出的比例。 $R = \frac{TP}{TP + FN}$

我们使用 **IoU (Intersection over Union)** 来判定预测框是否正确。如果 `IoU(Pred, GT) >= 阈值`，则视为 TP (True Positive)。

### 主要指标

| 指标键名 | 含义 | 详细解释 |
| :--- | :--- | :--- |
| **mAP** | **平均 AP** | **最重要的综合指标**。它是在 IoU 阈值从 0.50 到 0.95（步长 0.05）共 10 个阈值下计算出的 AP 的平均值，并且对所有类别取平均。它反映了模型在不同严格程度下的综合表现。在 COCO 竞赛中简称为 AP。 |
| **mAP_50** | IoU=0.50 的 mAP | PASCAL VOC 竞赛的标准指标。只要求 IoU > 0.5 就算正确，相对宽松。 |
| **mAP_75** | IoU=0.75 的 mAP | 严格模式下的指标。要求检测框与真值重叠度很高才算正确。 |
| **AP_<class_id>** | 特定类别的 AP | 该类别的 Average Precision (IoU 0.5:0.95)。 |
| **AP_50_<class_id>** | 特定类别的 AP@50 | 该类别在 IoU=0.50 下的 AP。 |
| **AP_75_<class_id>** | 特定类别的 AP@75 | 该类别在 IoU=0.75 下的 AP。 |

### 不同尺度的 AP

为了分析模型对不同大小物体的检测能力，我们将物体按面积（像素数）分为三类：

| 指标键名 | 含义 | 定义 (面积 = 宽 × 高) |
| :--- | :--- | :--- |
| **mAP_s** | 小目标 mAP | 面积 < $32^2$ (小于 1024 像素) |
| **mAP_m** | 中目标 mAP | $32^2 \le$ 面积 < $96^2$ (1024 ~ 9216 像素) |
| **mAP_l** | 大目标 mAP | 面积 $\ge 96^2$ (大于 9216 像素) |

通常小目标检测 (mAP_s) 是最难的。

### 平均召回率 (Average Recall, AR)

AR 反映了模型发现目标的能力，不考虑分数的排序，主要关注有多少真值被覆盖了。

EasyMetrics 按照每张图片允许的最大检测数 (Max Dets) 来计算 AR：

| 指标键名 | 含义 | 详细解释 |
| :--- | :--- | :--- |
| **AR_1** | MaxDets=1 的 AR | 每张图只取置信度最高的 1 个框来计算召回率。 |
| **AR_10** | MaxDets=10 的 AR | 每张图取前 10 个框计算召回率。 |
| **AR_100** | MaxDets=100 的 AR | 每张图取前 100 个框计算召回率。通常接近模型能达到的召回率上限。 |

### 计算细节

1. **101点插值法**:
   我们在计算 AP 时，采用 COCO 标准的 101 点插值法。即在 recall = [0.00, 0.01, ..., 1.00] 这 101 个点上取对应的最大 precision 并求平均。这比传统的 11 点插值法更精确。

2. **多尺度评估逻辑**:
   - 计算 `mAP_s` 时，我们只保留真实面积在 `[0, 32^2)` 范围内的 GT，并且只考虑面积在该范围内的预测框。
   - 如果某张图片没有该尺度的目标，则不参与该尺度的计算。

### 最佳阈值推荐 (Best Score Suggestion)

EasyMetrics 支持根据指定的精度 (Precision) 要求，自动寻找最佳的置信度阈值。这对于实际部署模型时设定阈值非常有帮助。

| 指标键名 | 含义 | 详细解释 |
| :--- | :--- | :--- |
| **BestScore_IoU0.50_P0.90_<class_id>** | 满足精度要求的最低阈值 | 在 IoU=0.50 的评估标准下，为了让该类别的预测精度达到 90%，建议设置的最低置信度阈值。 |

### 如何选择评估指标

- **如果只看一个数**: 关注 **mAP**。
- **如果定位精度很重要**: 关注 **mAP_75**。
- **如果漏检代价很大** (如自动驾驶): 关注 **AR_100**。
- **如果发现小物体检测很差**: 关注 **mAP_s**，可能需要调整数据增强或模型结构。

## � 扩展与定制

### 添加新任务

要添加新的评估任务（如分类、分割等）：

1. 在 `easyMetrics/tasks/` 下创建新目录
2. 创建评估接口文件
3. 继承 `easyMetrics.core.Metric` 基类
4. 实现 `reset()`, `update()` 和 `compute()` 方法

### 自定义匹配策略

EasyMetrics 支持自定义匹配策略，详情请参考源码中的 `matcher.py` 文件。

## 🎯 版本历史

### v0.4.3 (2026-02-06)
- **统一接口设计**: 简化为三个核心接口
  - `evaluate`: 统一评测接口，支持自动检测任务类型
  - `evaluate_detection`: 检测任务专用接口
  - `evaluate_classification`: 分类任务专用接口
- **精简代码结构**: 移除冗余接口，优化核心实现
- **增强自动检测能力**: 提高任务类型检测的准确性
  - 支持更多输入格式的自动识别
  - 为不同任务类型提供更智能的默认参数
- **更新文档**: 移除简化API用法部分，只保留核心接口示例
- **保持向后兼容**: 所有原有功能保持不变

### v0.4.2 (2026-02-06)
- **扩展AUC指标**: 支持多分类和多标签场景
  - 多分类: 支持 One-vs-Rest (OvR) 和 One-vs-One (OvO) 方法
  - 多标签: 为每个标签计算AUC并支持不同的平均方式
  - 支持宏平均、微平均和加权平均
- **扩展F1 Score指标**: 支持多标签场景
- **新增统一评估接口**: 实现 `evaluate` 函数作为统一评估入口
  - 自动判断任务类型（检测或分类）
  - 支持同时评估检测和分类指标
  - 提供统一的参数接口
- **完善示例代码**: 在 demo.py 中添加多分类和多标签的测试示例

### v0.4.1 (2026-02-06)
- **新增分类评估指标**:
  - F1 Score: 支持二分类和多分类场景，包括宏平均、微平均和加权平均
  - AUC: 支持二分类场景的ROC曲线下面积计算，包括线性插值和梯形积分方法
- **扩展模块结构**: 新增 classification 任务模块
- **完善导出配置**: 更新模块导出，支持直接从 easyMetrics 导入分类指标
- **添加示例代码**: 在 demo.py 中添加分类指标的完整测试示例

### v0.4.0 (2026-02-05)
- **增强批量处理能力**: 支持批量 YOLO 格式数据的处理
- **修复标签处理**: 完善对二维标签数组的处理
- **提升鲁棒性**: 增强对各种边缘情况的处理能力
- **优化性能**: 改进并行计算效率
- **完善文档**: 添加批量 YOLO 格式数据示例

### v0.3.0 (2026-02-05)
- **增强类型兼容性**: 支持标量、列表、numpy 数组等多种输入类型
- **修复边界情况**: 优化空数据、只有预测/目标等特殊情况的处理
- **提升稳定性**: 修复索引错误和类型转换错误
- **增强格式支持**: 完善对不同输入格式的处理
- **改进用户体验**: 添加进度条功能，可选择禁用
- **完善文档**: 更新使用指南和指标详解

### v0.2.0 (2026-02-05)
- **修复类型转换**: 解决标量输入的处理问题
- **提升稳定性**: 优化类型转换逻辑
- **添加格式支持**: 支持 COCO、VOC、YOLO 等多种输入格式
- **添加并行计算**: 支持多核并行加速

### v0.1.0 (2026-02-05)
- **初始版本**: 发布核心功能
- **目标检测评估**: 支持完整的 COCO 指标
- **最佳阈值计算**: 支持自动计算最佳置信度阈值
- **基础格式支持**: 支持 COCO 格式输入

## 💡 实用技巧

1. **数据预处理**：确保输入的边界框坐标格式正确（`[x1, y1, x2, y2]`），且为浮点数类型。

2. **类别索引**：确保预测结果和真实标签使用相同的类别索引体系。

3. **性能优化**：
   - 对于大规模数据集，使用 `n_jobs=-1` 启用并行计算
   - 使用 `metrics` 参数只计算需要的指标

4. **结果分析**：
   - 关注 `mAP_50` 指标评估模型在宽松条件下的性能
   - 关注 `mAP_75` 指标评估模型在严格条件下的性能
   - 使用 `AR_100` 评估模型的召回能力
   - 使用 `mAP_s`/`mAP_m`/`mAP_l` 分析模型对不同尺度目标的表现

5. **格式选择**：
   - 对于标准场景，使用默认的 COCO 格式
   - 对于与其他框架集成，可选择 VOC 或 YOLO 格式
   - 对于特殊需求，可使用自定义格式转换函数

## 🤝 贡献指南

欢迎贡献代码和提出建议！

1. **Fork 项目**
2. **创建特性分支**
3. **提交更改**
4. **发起 Pull Request**

### 开发规范

- **代码风格**：遵循 PEP 8 规范
- **测试**：为新功能添加测试用例
- **文档**：更新相关文档
- **提交信息**：清晰、简洁的提交信息

## 📄 许可证

EasyMetrics 使用 MIT 许可证，详见 LICENSE 文件。

## 📞 联系我们

如有问题或建议，欢迎通过 GitHub Issues 提出。

---

*Created with ❤️ by EasyMetrics Team*
