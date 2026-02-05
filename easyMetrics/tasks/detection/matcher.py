from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import numpy as np

class BaseMatcher(ABC):
    """
    匹配策略的抽象基类。
    定义如何将预测框与真实框进行匹配。
    """
    
    @abstractmethod
    def match(self, 
              class_preds: List[Tuple], 
              class_gt: Dict, 
              pred_gt_ious: List[np.ndarray], 
              iou_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行匹配逻辑。
        
        参数:
            class_preds: 预测列表，每个元素为 (score, img_idx, box, rank)
            class_gt: GT 字典，key 为 img_idx，value 为 {'boxes': [], 'used': []}
            pred_gt_ious: 预计算的 IoU 列表，对应 class_preds 中的每个预测
            iou_thresh: 当前的 IoU 阈值
            
        返回:
            tp: (N,) 数组，1 表示 True Positive
            fp: (N,) 数组，1 表示 False Positive
        """
        pass

class GreedyIoUMatcher(BaseMatcher):
    """
    标准的贪婪 IoU 匹配策略 (COCO/VOC 标准)。
    按分数从高到低，优先匹配 IoU 最大的 GT。
    """
    
    def match(self, 
              class_preds: List[Tuple], 
              class_gt: Dict, 
              pred_gt_ious: List[np.ndarray], 
              iou_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        # 重置 GT 使用状态
        # 注意：这里会修改 class_gt 的内部状态，这是预期的副作用
        for img_data in class_gt.values():
            img_data['used'][:] = False
            
        for i, ious in enumerate(pred_gt_ious):
            if len(ious) == 0:
                fp[i] = 1
                continue
            
            # 找到最佳匹配
            best_gt_idx = np.argmax(ious)
            best_iou = ious[best_gt_idx]
            
            img_idx = class_preds[i][1]
            
            if best_iou >= iou_thresh:
                # 检查该 GT 是否已被匹配
                if not class_gt[img_idx]['used'][best_gt_idx]:
                    tp[i] = 1
                    class_gt[img_idx]['used'][best_gt_idx] = True
                else:
                    fp[i] = 1 # 重复检测 (Duplicate)
            else:
                fp[i] = 1 # IoU 不足
                
        return tp, fp
