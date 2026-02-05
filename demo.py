import numpy as np
from easyMetrics.tasks.detection import evaluate_detection

def main():
    print("=== EasyMetrics 详细示例 ===")

    # 1. 准备数据
    # 格式: [x1, y1, x2, y2]
    # 假设有 2 张图片
    
    # 图片 1: 预测完全正确
    preds_1 = {
        'boxes': np.array([[10, 10, 50, 50]], dtype=float),
        'scores': np.array([0.95], dtype=float),
        'labels': np.array([0], dtype=int)
    }
    targets_1 = {
        'boxes': np.array([[10, 10, 50, 50]], dtype=float),
        'labels': np.array([0], dtype=int)
    }

    # 图片 2: 有一个误检 (FP) 和一个漏检 (FN)
    # 预测: 
    # - [100, 100, 150, 150] (IoU=0.9, 正确)
    # - [0, 0, 20, 20] (误检)
    preds_2 = {
        'boxes': np.array([
            [100, 100, 150, 150], 
            [0, 0, 20, 20]
        ], dtype=float),
        'scores': np.array([0.9, 0.6], dtype=float),
        'labels': np.array([0, 0], dtype=int)
    }
    targets_2 = {
        'boxes': np.array([
            [100, 100, 150, 150],
            [200, 200, 250, 250] # 这个没被检测到 (漏检)
        ], dtype=float),
        'labels': np.array([0, 0], dtype=int)
    }

    preds = [preds_1, preds_2]
    targets = [targets_1, targets_2]

    # 2. 基本用法 - 计算标准 COCO 指标
    print("\n[1] 基本用法 - 计算标准 COCO 指标...")
    results = evaluate_detection(preds, targets)
    
    print(f"mAP (IoU 0.5:0.95): {results['mAP']:.4f}")
    print(f"mAP_50 (IoU 0.5)  : {results['mAP_50']:.4f}")
    print(f"mAP_75 (IoU 0.75) : {results['mAP_75']:.4f}")
    print(f"AR_100 (MaxDets=100): {results['AR_100']:.4f}")

    # 3. 并行计算示例
    print("\n[2] 并行计算示例...")
    print("使用 4 个核心进行评估:")
    results_parallel = evaluate_detection(preds, targets, n_jobs=4)
    print(f"mAP: {results_parallel['mAP']:.4f}")
    print("并行计算测试完成")

    # 4. 自定义指标筛选示例
    print("\n[3] 自定义指标筛选示例...")
    print("只计算 mAP 和 mAP_50 指标:")
    results_custom = evaluate_detection(preds, targets, metrics=['mAP', 'mAP_50'])
    print(f"自定义指标结果: {results_custom}")
    print("自定义指标筛选测试完成")

    # 5. 寻找最佳阈值
    # 场景: 我们希望在 IoU=0.5 的情况下，精度(Precision)至少达到 90%。
    # 问: 我应该设置多高的置信度阈值？
    print("\n[4] 寻找最佳置信度阈值...")
    results_criteria = evaluate_detection(
        preds, targets,
        score_criteria=[(0.5, 0.9)] # (IoU, MinPrecision)
    )
    
    best_thresh = results_criteria.get('BestScore_IoU0.50_P0.90_0')
    print(f"类别 0 在 IoU=0.5 下满足 P>=0.9 的推荐阈值: {best_thresh}")

    # 6. 进度条控制示例
    print("\n[5] 进度条控制示例...")
    print("禁用进度条进行评估:")
    results_no_progress = evaluate_detection(preds, targets, progress=False)
    print(f"mAP: {results_no_progress['mAP']:.4f}")
    print("进度条控制测试完成")

    # 7. 多类别评估示例
    print("\n[6] 多类别评估示例...")
    # 准备多类别数据
    preds_multi = [{
        'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float),
        'scores': np.array([0.95, 0.9], dtype=float),
        'labels': np.array([0, 1], dtype=int)  # 两个不同的类别
    }]
    targets_multi = [{
        'boxes': np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float),
        'labels': np.array([0, 1], dtype=int)  # 两个不同的类别
    }]
    
    results_multi = evaluate_detection(preds_multi, targets_multi)
    print(f"多类别评估 mAP: {results_multi['mAP']:.4f}")
    print(f"类别 0 的 AP: {results_multi['AP_0']:.4f}")
    print(f"类别 1 的 AP: {results_multi['AP_1']:.4f}")
    print("多类别评估测试完成")

    # 8. 不同格式输入示例
    print("\n[7] 不同格式输入示例...")
    # VOC 格式预测
    preds_voc = [[10, 10, 50, 50, 0, 0.95]]  # [x1, y1, x2, y2, class_id, confidence]
    # VOC 格式真值
    targets_voc = [[10, 10, 50, 50, 0]]  # [x1, y1, x2, y2, class_id]
    
    results_voc = evaluate_detection(
        [preds_voc], [targets_voc], 
        pred_format="voc", 
        target_format="voc"
    )
    print(f"VOC 格式输入评估 mAP: {results_voc['mAP']:.4f}")
    print("不同格式输入测试完成")

    print("\n=== 所有示例测试完成！===")

if __name__ == "__main__":
    main()
