from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class DetectionFormatConverter:
    """
    检测格式转换器，支持多种常见的目标检测格式转换为内部统一格式。
    
    支持的格式：
    - coco: COCO 格式，{"boxes": [[x1, y1, x2, y2]], "scores": [0.9], "labels": [0]}
    - voc: VOC 格式，支持列表形式 [[x1, y1, x2, y2, class_id]]
    - yolo: YOLO 格式，[[class_id, x, y, w, h, confidence]]，其中 x,y,w,h 是归一化值
    - custom: 自定义格式，需要提供转换函数
    """
    
    @staticmethod
    def convert(
        preds: List[Any], 
        targets: List[Any], 
        format: str = "coco",
        pred_format: Optional[str] = None,
        target_format: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        将输入数据转换为内部统一格式。
        
        参数:
            preds: 预测结果，可以是不同格式的数据。
            targets: 真实标签，可以是不同格式的数据。
            format: 输入数据的格式，当 pred_format 和 target_format 未指定时使用。
                    支持 "coco", "voc", "yolo", "custom"。
            pred_format: 预测结果的格式，优先级高于 format。
            target_format: 真实标签的格式，优先级高于 format。
            **kwargs: 额外的转换参数。
                - image_size: YOLO 格式需要的图像尺寸 (width, height)
                - custom_converter: 自定义格式的转换函数
                
        返回:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: 转换后的预测结果和真实标签。
        """
        # 确定最终使用的格式
        final_pred_format = pred_format or format
        final_target_format = target_format or format
        
        # 转换预测结果
        if final_pred_format == "coco":
            converted_preds = DetectionFormatConverter._convert_coco_preds(preds)
        elif final_pred_format == "voc":
            converted_preds = DetectionFormatConverter._convert_voc_preds(preds)
        elif final_pred_format == "yolo":
            image_size = kwargs.get("image_size", (640, 640))
            converted_preds = DetectionFormatConverter._convert_yolo_preds(preds, image_size)
        elif final_pred_format == "custom":
            custom_converter = kwargs.get("custom_converter")
            if not custom_converter:
                raise ValueError("custom format requires custom_converter function")
            # 对于自定义格式，先转换整体，然后提取预测部分
            temp_preds, _ = custom_converter(preds, [], **kwargs)
            converted_preds = temp_preds
        else:
            raise ValueError(f"Unsupported pred_format: {final_pred_format}")
        
        # 转换真实标签
        if final_target_format == "coco":
            converted_targets = DetectionFormatConverter._convert_coco_targets(targets)
        elif final_target_format == "voc":
            converted_targets = DetectionFormatConverter._convert_voc_targets(targets)
        elif final_target_format == "yolo":
            image_size = kwargs.get("image_size", (640, 640))
            converted_targets = DetectionFormatConverter._convert_yolo_targets(targets, image_size)
        elif final_target_format == "custom":
            custom_converter = kwargs.get("custom_converter")
            if not custom_converter:
                raise ValueError("custom format requires custom_converter function")
            # 对于自定义格式，先转换整体，然后提取标签部分
            _, temp_targets = custom_converter([], targets, **kwargs)
            converted_targets = temp_targets
        else:
            raise ValueError(f"Unsupported target_format: {final_target_format}")
        
        return converted_preds, converted_targets
    
    @staticmethod
    def _convert_coco_preds(coco_preds: List[Any]) -> List[Dict[str, Any]]:
        """
        转换 COCO 格式的预测结果。
        
        支持两种格式：
        1. COCO 官方格式: [{"image_id": 0, "category_id": 0, "bbox": [0, 0, 100, 100], "score": 0.9}]
        2. 内部统一格式: [{"boxes": [[x1, y1, x2, y2]], "scores": [0.9], "labels": [0]}]
        """
        # 检查是否已经是内部统一格式
        if coco_preds and isinstance(coco_preds[0], dict):
            first_pred = coco_preds[0]
            if 'boxes' in first_pred and 'scores' in first_pred and 'labels' in first_pred:
                # 已经是内部统一格式，直接返回
                return coco_preds
        
        # 按 image_id 分组
        grouped_preds = {}
        for pred in coco_preds:
            image_id = pred.get('image_id', 0)
            if image_id not in grouped_preds:
                grouped_preds[image_id] = {
                    "boxes": [],
                    "scores": [],
                    "labels": []
                }
            # COCO bbox 格式: [x, y, width, height]，转换为 [x1, y1, x2, y2]
            x, y, width, height = pred['bbox']
            boxes = [x, y, x + width, y + height]
            grouped_preds[image_id]["boxes"].append(boxes)
            grouped_preds[image_id]["scores"].append(pred['score'])
            grouped_preds[image_id]["labels"].append(pred['category_id'])
        
        # 转换为列表格式
        converted = list(grouped_preds.values())
        return converted
    
    @staticmethod
    def _convert_coco_targets(coco_targets: List[Any]) -> List[Dict[str, Any]]:
        """
        转换 COCO 格式的真实标签。
        
        支持两种格式：
        1. COCO 官方格式: [{"image_id": 0, "category_id": 0, "bbox": [0, 0, 100, 100], "area": 10000, "iscrowd": 0}]
        2. 内部统一格式: [{"boxes": [[x1, y1, x2, y2]], "labels": [0]}]
        """
        # 检查是否已经是内部统一格式
        if coco_targets and isinstance(coco_targets[0], dict):
            first_target = coco_targets[0]
            if 'boxes' in first_target and 'labels' in first_target:
                # 已经是内部统一格式，直接返回
                return coco_targets
        
        # 按 image_id 分组
        grouped_targets = {}
        for target in coco_targets:
            image_id = target.get('image_id', 0)
            if image_id not in grouped_targets:
                grouped_targets[image_id] = {
                    "boxes": [],
                    "labels": []
                }
            # COCO bbox 格式: [x, y, width, height]，转换为 [x1, y1, x2, y2]
            x, y, width, height = target['bbox']
            boxes = [x, y, x + width, y + height]
            grouped_targets[image_id]["boxes"].append(boxes)
            grouped_targets[image_id]["labels"].append(target['category_id'])
        
        # 转换为列表格式
        converted = list(grouped_targets.values())
        return converted
    
    @staticmethod
    def _convert_voc_preds(voc_preds: List[Any]) -> List[Dict[str, Any]]:
        """
        转换 VOC 格式的预测结果。
        VOC 格式示例: [[x1, y1, x2, y2, class_id, confidence]]
        """
        converted = []
        for pred in voc_preds:
            if isinstance(pred, list):
                # 单张图片的预测
                boxes = []
                scores = []
                labels = []
                for item in pred:
                    if len(item) >= 6:
                        # [x1, y1, x2, y2, class_id, confidence]
                        boxes.append(item[:4])
                        scores.append(item[5])
                        labels.append(int(item[4]))
                    elif len(item) == 5:
                        # [x1, y1, x2, y2, class_id]
                        boxes.append(item[:4])
                        scores.append(1.0)  # 默认置信度
                        labels.append(int(item[4]))
                converted.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                })
            elif isinstance(pred, dict):
                # 已经是 COCO 格式
                converted.append(pred)
        return converted
    
    @staticmethod
    def _convert_voc_targets(voc_targets: List[Any]) -> List[Dict[str, Any]]:
        """
        转换 VOC 格式的真实标签。
        VOC 格式示例: [[x1, y1, x2, y2, class_id]]
        """
        converted = []
        for target in voc_targets:
            if isinstance(target, list):
                # 单张图片的标签
                boxes = []
                labels = []
                for item in target:
                    if len(item) >= 5:
                        # [x1, y1, x2, y2, class_id]
                        boxes.append(item[:4])
                        labels.append(int(item[4]))
                converted.append({
                    "boxes": boxes,
                    "labels": labels
                })
            elif isinstance(target, dict):
                # 已经是 COCO 格式
                converted.append(target)
        return converted
    
    @staticmethod
    def _convert_yolo_preds(yolo_preds: List[Any], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        转换 YOLO 格式的预测结果。
        YOLO 格式示例: [[class_id, x, y, w, h, confidence]]
        其中 x, y, w, h 是归一化值 (0-1)
        """
        converted = []
        width, height = image_size
        
        for pred in yolo_preds:
            if isinstance(pred, list):
                # 单张图片的预测
                boxes = []
                scores = []
                labels = []
                for item in pred:
                    if len(item) >= 6:
                        # [class_id, x, y, w, h, confidence]
                        class_id, x, y, w, h, conf = item
                        # 转换为 [x1, y1, x2, y2]
                        x1 = (x - w/2) * width
                        y1 = (y - h/2) * height
                        x2 = (x + w/2) * width
                        y2 = (y + h/2) * height
                        boxes.append([x1, y1, x2, y2])
                        scores.append(conf)
                        labels.append(int(class_id))
                    elif len(item) == 5:
                        # [class_id, x, y, w, h]
                        class_id, x, y, w, h = item
                        # 转换为 [x1, y1, x2, y2]
                        x1 = (x - w/2) * width
                        y1 = (y - h/2) * height
                        x2 = (x + w/2) * width
                        y2 = (y + h/2) * height
                        boxes.append([x1, y1, x2, y2])
                        scores.append(1.0)  # 默认置信度
                        labels.append(int(class_id))
                converted.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                })
            elif isinstance(pred, dict):
                # 已经是 COCO 格式
                converted.append(pred)
        return converted
    
    @staticmethod
    def _convert_yolo_targets(yolo_targets: List[Any], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        转换 YOLO 格式的真实标签。
        YOLO 格式示例: [[class_id, x, y, w, h]]
        其中 x, y, w, h 是归一化值 (0-1)
        """
        converted = []
        width, height = image_size
        
        for target in yolo_targets:
            if isinstance(target, list):
                # 单张图片的标签
                boxes = []
                labels = []
                for item in target:
                    if len(item) >= 5:
                        # [class_id, x, y, w, h]
                        class_id, x, y, w, h = item[:5]
                        # 转换为 [x1, y1, x2, y2]
                        x1 = (x - w/2) * width
                        y1 = (y - h/2) * height
                        x2 = (x + w/2) * width
                        y2 = (y + h/2) * height
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(class_id))
                converted.append({
                    "boxes": boxes,
                    "labels": labels
                })
            elif isinstance(target, dict):
                # 已经是 COCO 格式
                converted.append(target)
        return converted
