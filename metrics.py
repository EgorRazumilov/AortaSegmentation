import torch

SMOOTH = 1e-6


class IoUMetricOneClass:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)

        outputs = torch.sigmoid(outputs) > self.threshold
        labels = labels > self.threshold

        intersection = (outputs & labels).float().sum((1, 2))
        union = (outputs | labels).float().sum((1, 2))

        iou = (intersection + SMOOTH) / (union + SMOOTH)
        return iou.mean()

class DiceMetric:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)

        outputs = torch.sigmoid(outputs) > self.threshold
        labels = labels > self.threshold

        intersection = (outputs & labels).float().sum((1, 2))
        union = (outputs).float().sum((1, 2)) + (labels).float().sum((1, 2))

        dice = (2 * intersection + SMOOTH) / (union + SMOOTH)
        return dice.mean()


class IoUMetricOneClass3D:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)

        outputs = torch.sigmoid(outputs) > self.threshold
        labels = labels > self.threshold

        intersection = (outputs & labels).float().sum((1, 2, 3))
        union = (outputs | labels).float().sum((1, 2, 3))

        iou = (intersection + SMOOTH) / (union + SMOOTH)
        return iou.mean()

class DiceMetric3D:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)

        outputs = torch.sigmoid(outputs) > self.threshold
        labels = labels > self.threshold

        intersection = (outputs & labels).float().sum((1, 2, 3))
        union = (outputs).float().sum((1, 2, 3)) + (labels).float().sum((1, 2, 3))

        dice = (2 * intersection + SMOOTH) / (union + SMOOTH)
        return dice.mean()
