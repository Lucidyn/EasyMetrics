import numpy as np

def calculate_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算两组边界框之间的交并比 (IoU)。
    
    参数:
        boxes1: (N, 4) ndarray, 格式为 [x1, y1, x2, y2]
        boxes2: (M, 4) ndarray, 格式为 [x1, y1, x2, y2]
        
    返回:
        iou: (N, M) ndarray, 表示 boxes1 和 boxes2 之间的重叠程度
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))

    # 确保输入是 2D 的
    if boxes1.ndim == 1:
        boxes1 = boxes1[np.newaxis, :]
    if boxes2.ndim == 1:
        boxes2 = boxes2[np.newaxis, :]

    # 计算面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # 广播计算交集区域的左上角和右下角坐标
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2) [x1, y1]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2) [x2, y2]

    # 计算交集宽高，clip(0) 确保无重叠时为 0
    wh = np.clip(rb - lt, 0, None)  # (N, M, 2) [w, h]
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # 计算并集面积
    union = area1[:, None] + area2[None, :] - inter

    # 避免除以零
    union = np.maximum(union, 1e-6)

    iou = inter / union
    return iou

def compute_ap_coco(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    使用 COCO 风格的 101 点插值法计算平均精度 (Average Precision)。
    
    参数:
        recall: (N,) ndarray, 召回率数组，需单调递增
        precision: (N,) ndarray, 对应的精度数组
        
    返回:
        ap: float, 计算得到的 AP 值
    """
    # 在开头和结尾添加哨兵值
    # 注意: COCO 不严格要求开头为 0，但要求覆盖 [0, 1] 区间
    # 我们采用类似的包络线方法，但进行固定采样
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # 计算精度包络线 (单调递减)
    # 对于每个 recall 值，取其右侧最大的 precision
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 生成从 0.0 到 1.00 的 101 个召回率阈值
    rec_thresholds = np.linspace(0.0, 1.00, 101)
    
    # 对于每个阈值 t，我们需要找到 recall >= t 时的最大 precision
    # 由于 mpre[i] 已经是 recall >= mrec[i] 时的最大 precision，
    # 我们只需要找到 mrec 中第一个大于等于 t 的位置。
    inds = np.searchsorted(mrec, rec_thresholds, side='left')
    
    # 获取包络线上的值
    q = mpre[inds]
    
    return float(np.mean(q))
