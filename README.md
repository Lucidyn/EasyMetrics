# EasyMetrics

一个轻量级、零依赖的机器学习指标评估平台，基于 `numpy` 从零构建，专注于提供简单易用且准确的模型评估工具。

## ✨ 核心特性
- **零依赖**: 仅需 Python 和 Numpy，无需安装大型深度学习框架
- **易于扩展**: 模块化设计，通过继承 `Metric` 基类即可添加新任务
- **功能强大**: 完美支持目标检测任务的全方位评估
  - 标准 COCO 指标: mAP、mAP_50、mAP_75、mAP_s/m/l
  - 平均召回率 (AR) 指标
  - 每类别独立评估
  - **独家功能**: 自动计算满足特定精度要求的最佳置信度阈值

## � 目录结构
```
easyMetrics/
├── easyMetrics/         # 核心代码
│   ├── core/             # 抽象基类
│   │   └── base.py
│   └── tasks/            # 任务实现
│       └── detection/    # 目标检测
│           ├── interface.py # 对外接口
│           ├── map.py     # mAP 核心逻辑
│           ├── matcher.py # 匹配策略
│           ├── utils.py   # 辅助函数
│           └── format_converter.py # 格式转换器
├── docs/                 # 文档
│   ├── 使用指南.md
│   └── 指标详解.md
├── demo.py               # 使用示例
└── README.md
```

## 🚀 快速上手

### 目标检测评估

使用 `evaluate_detection` 函数，一行代码完成评估：

```python
import numpy as np
from easyMetrics.tasks.detection import evaluate_detection

# 准备数据 - 每张图片一个字典
preds = [{
    'boxes': np.array([[10, 10, 50, 50]]),  # [x1, y1, x2, y2] 格式
    'scores': np.array([0.9]),              # 置信度分数
    'labels': np.array([0])                 # 类别索引
}]
targets = [{
    'boxes': np.array([[10, 10, 50, 50]]),  # 真实边界框
    'labels': np.array([0])                 # 真实类别
}]

# 1. 计算标准 COCO 指标
results = evaluate_detection(preds, targets)
print(f"mAP: {results['mAP']:.4f}")
print(f"mAP_50: {results['mAP_50']:.4f}")

# 2. 寻找最佳置信度阈值
# 场景: IoU=0.5 时精度至少达到 90%
results = evaluate_detection(
    preds, targets, 
    score_criteria=[(0.5, 0.9)]
)
print(f"推荐阈值: {results.get('BestScore_IoU0.50_P0.90_0')}")
```

### 并行加速

对于大规模数据集，启用多核并行计算：

```python
# 使用 4 个核心
results = evaluate_detection(preds, targets, n_jobs=4)

# 使用所有可用核心
results = evaluate_detection(preds, targets, n_jobs=-1)
```

## 🔧 扩展新任务

添加新指标（例如分类任务的准确率）：

1. 在 `easyMetrics/tasks/` 下创建新目录（如 `classification`）
2. 继承 `easyMetrics.core.Metric` 基类
3. 实现 `reset()`, `update()` 和 `compute()` 方法

---
*Created with ❤️ by EasyMetrics Team*
