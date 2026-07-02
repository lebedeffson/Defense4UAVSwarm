from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _model_device(yolo, device: str | None) -> torch.device:
    model_device = next(yolo.parameters()).device
    if device and device != "auto":
        model_device = torch.device(device)
        yolo.to(model_device)
    return model_device


def _tensor_output(pred):
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if isinstance(pred, dict):
        tensors = [v for v in pred.values() if torch.is_tensor(v)]
        if not tensors:
            raise RuntimeError("YOLO forward returned no tensor outputs for FGSM")
        pred = tensors[0]
    if pred.ndim < 2:
        raise RuntimeError(f"Unexpected YOLO output shape for FGSM: {tuple(pred.shape)}")
    return pred


def _scores(pred: torch.Tensor) -> torch.Tensor:
    return pred[:, 4:, :] if pred.ndim == 3 and pred.shape[1] > 5 else pred


def _gt_boxes_tensor(gt_boxes, shape: tuple[int, int], imgsz: int, device: torch.device) -> torch.Tensor | None:
    if gt_boxes is None or len(gt_boxes) == 0:
        return None
    h, w = shape
    arr = np.asarray(gt_boxes, dtype=np.float32).copy()
    arr[:, [0, 2]] *= imgsz / max(1, w)
    arr[:, [1, 3]] *= imgsz / max(1, h)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=1)


def _max_iou(box: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ix1 = torch.maximum(box[0], gt[:, 0])
    iy1 = torch.maximum(box[1], gt[:, 1])
    ix2 = torch.minimum(box[2], gt[:, 2])
    iy2 = torch.minimum(box[3], gt[:, 3])
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    b_area = torch.clamp(box[2] - box[0], min=0) * torch.clamp(box[3] - box[1], min=0)
    g_area = torch.clamp(gt[:, 2] - gt[:, 0], min=0) * torch.clamp(gt[:, 3] - gt[:, 1], min=0)
    return (inter / torch.clamp(b_area + g_area - inter, min=1e-6)).max()


def fgsm_loss(pred: torch.Tensor, loss_mode: str, gt_boxes=None, beta: float = 1.0) -> tuple[torch.Tensor, str]:
    pred = _tensor_output(pred)
    scores = _scores(pred).float()
    conf_loss = -scores.max()
    if loss_mode != "proxy_conf_box_if_available":
        return conf_loss, "class_only"
    if gt_boxes is None or pred.ndim != 3 or pred.shape[1] < 5:
        return conf_loss, "class_only_fallback"
    det_scores = scores[0].max(dim=0).values
    best_idx = int(det_scores.argmax().detach().cpu())
    pred_box = _xywh_to_xyxy(pred[0, :4, :].float().T)[best_idx]
    iou_loss = -_max_iou(pred_box, gt_boxes)
    return conf_loss + beta * iou_loss, "proxy_conf_box_if_available"


def fgsm_image_with_diagnostics(
    image_path: Path,
    eps: float,
    out_path: Path,
    yolo,
    imgsz: int,
    device: str | None,
    loss_mode: str = "class_only",
    gt_boxes=None,
    beta: float = 1.0,
    force: bool = False,
) -> tuple[Path, dict]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    model_device = _model_device(yolo, device)
    x = x.to(model_device)
    gt_t = _gt_boxes_tensor(gt_boxes, (h, w), imgsz, model_device)

    yolo.eval()
    if out_path.exists() and not force:
        return out_path, {}

    x.requires_grad_(True)
    pred = yolo(x)
    loss, actual_loss_mode = fgsm_loss(pred, loss_mode, gt_t, beta)
    yolo.zero_grad(set_to_none=True)
    loss.backward()
    if x.grad is None:
        raise RuntimeError("No input gradient for FGSM")
    grad = x.grad.detach()
    perturb_small_t = grad.sign() * eps
    x_adv_small = torch.clamp(x.detach() + perturb_small_t, 0.0, 1.0)
    with torch.no_grad():
        loss_adv, _ = fgsm_loss(yolo(x_adv_small), loss_mode, gt_t, beta)

    perturb_small = perturb_small_t[0].permute(1, 2, 0).cpu().numpy()
    perturb = cv2.resize(perturb_small, (w, h), interpolation=cv2.INTER_NEAREST)
    clean = rgb.astype(np.float32) / 255.0
    adv = np.clip(clean + perturb, 0.0, 1.0)
    real_perturb = adv - clean

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor((adv * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    diag = {
        "loss_clean": float(loss.detach().cpu()),
        "loss_adv": float(loss_adv.detach().cpu()),
        "loss_mode_actual": actual_loss_mode,
        "grad_l1": float(grad.abs().mean().cpu()),
        "grad_l2": float(torch.linalg.vector_norm(grad).cpu()),
        "grad_linf": float(grad.abs().max().cpu()),
        "perturb_linf": float(np.abs(real_perturb).max()),
        "perturb_l2": float(np.linalg.norm(real_perturb.reshape(-1))),
        "image_min_clean": float(clean.min()),
        "image_max_clean": float(clean.max()),
        "image_min_adv": float(adv.min()),
        "image_max_adv": float(adv.max()),
    }
    return out_path, diag


def fgsm_class_only_image(image_path: Path, eps: float, out_path: Path, yolo, imgsz: int, device: str | None) -> Path:
    """Backward-compatible class-only FGSM cache writer."""
    if out_path.exists():
        return out_path
    path, _ = fgsm_image_with_diagnostics(image_path, eps, out_path, yolo, imgsz, device, loss_mode="class_only")
    return path
