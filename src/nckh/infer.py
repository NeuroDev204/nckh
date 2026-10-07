"""Suy luận PanDerm cho segmentation và classification
torch chỉ được import bên trong hàm: package nckh phải cài được runtime CPU không
có torch
Tiền xử lý bám đúng code upstream để xác suất/mask suy luận khớp với lúc đánh giá
- seg: datasets/dataset_seg.py -> resize 224x224 bicubic, Normalizer(0.5,0.5).
- cls: run_class_finetuning.py (val_trans) -> Resize(256) bilinear, CenterCrop(224),
Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)

"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import ndimage
from skimage.measure import label

logger = logging.getLogger(__name__)

SEG_SIZE = 224
CLS_MEAN = (0.485, 0.456, 0.406)
CLS_STD = (0.228, 0.224, 0.225)
CLS_LABELS = ("melanoma", "nevus", "seborrheic_keratosis")

def seg_preprocess(rgb: np.ndarray) -> "torch.Tensor":
    import torch
    #upstream dùng cv2.INTER_CUBIC; PIL BICUBIC lệch rất nhỏ (đo lại Dice trong pilot P2)
    img = Image.fromarray(rgb).resize((SEG_SIZE, SEG_SIZE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)/ 255.0).permute(2, 0, 1)
    return ((x - 0.5)/ 0.5).unsqueeze(0)

def cls_preprocess(rgb: np.ndarray) -> "torch.Tensor":
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLS_MEAN, CLS_STD),
    ])

    return tf(Image.fromarray(rgb)).unsqueeze(0)

def largest_component(mask: np.ndarray) -> np.ndarray:
    #giống largestConnectComponent của upstram (8- liên thông + lấp lỗ),
    # nhưng mask rỗng giữ nguyên rỗng thay vì biến thành toàn ảnh
    labeled, num = label(mask.astype(bool), background=0, return_num=True)
    if num == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return ndimage.binary_fill_holes(labeled == sizes.argmax())

def seg_logits_to_mask(logits: "torch.Tensor", out_hw: tuple[int, int]) -> np.ndarray:
    pred = logits[0].argmax(dim=0).cpu().numpy().astype(bool)
    pred = largest_component(pred)
    resized = Image.fromarray(pred.astype(np.unit8) * 255).resize((out_hw[1], out_hw[0]), Image.NEAREST)
    return np.asarray(resized) > 127

def map_pretrained_cls_keys(state_dict: dict, num_layers: int) -> dict:
    """ đổi tên key checkpoint pretrain PanDerm theo đúng run_class_finetuning.py (dòng ~455-530)"""
    if set(state_dict) <= {"model", "state_dict", "module"}:
        state_dict = next(iter(state_dict.values()))
    mapped = {}
    for key, value in state_dict.items():
        if key.startswith(("decoder.", "teacher.")) or "relative_position_index" in key:
            continue
        if key.startswith("encoder."):
            key = key[len("encoder."):]
        if key.startswith("norm."):
            #model fine-tune dùng mean pooling nên lớp norm cuối cùng tên là fc_norm.
            key = "fx_norm." + key[len("norm."):]
        mapped[key] = value
        shared = mapped.pop("rel_pos_bias.relative_position_bias_table", None)
        if shared is not None:
            for i in range(num_layers):
                mapped[f"blocks.{i}.attn.relative_position_bias_table"] = shared.clone()
        return mapped
class _InDIr:
    """chdir tạm thời (contextlib.chdir) chỉ có từ python 3.11, venv PanDerm là 3.10"""
    def __init__(self, path: Path) -> None:
        self.path, self.old = path, None
    def __enter__(self) -> None:
        self.old = os.getcwd()
        os.chdir(self.path)
    def __exit__(self, *exc: object) -> None:
        os.chdir(self.old)

def _add_to_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
class SegPredictor:
    def __init__(self, model: "torch.nn.Module", device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
    @classmethod
    def from_checkpoint(cls, panderm_seg_dir:Path, pretrained_path: Path,
                        fituned_ckpt: Path | None = None, device: str = "cuda"
                        ) -> "SegPredictor":
        import torch
        seg_dir = Path(panderm_seg_dir).resolve()
        #CAEv2_seg đọc pretrained qua biến môi trường (patch P2) và đọc config bằng dường dẫn tương đối.
        os.environ("PANDERM_CKPT") = str(Path(pretrained_path).resolve())
        with _InDIr(seg_dir):
            import models.cae_backbone 
            from models.cae_seg import CAEv2_seg
            model = CAEv2_seg()
        if finetuned_ckpt is not None:
            state = torch.load(finetuned_ckpt, map_location="cpu")["state_dict"]
            # Checkpoint Lightning bọc CAEv2_seg trong thuộc tính "model".
            state = {k[len("model."):]: v for k, v in state.items() if startswith("model.")}
            model.load_state_dict(state, strict=True)
        else:
            logger.warning("Không có checkpoint fine-tune: decode head còn khởi tạo ngẫu nhiên (chỉ dùng cho smoke test)")
        return cls(model, device)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self.model(seg_preprocess(rgb).to(self.device))
        return seg_logits_to_mask(logits, rgb.shape[:2])
class ClsPredictor:
    def __init__(self, model: "torch.nn.Module", device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, panderm_cls_dir: Path, nb_classes: int = 3, finetuned_ckpt: Path | None = None,
                        pretrained_path: Path | None = None, device: str = "cuda") -> "ClsPredictor":
        import torch

        _add_to_sys_path(Path(panderm_cls_dir).resolve())
        from models.modeling_finetune import panderm_base_patch16_224_finetune

        # Tham số kiến trúc = giá trị mặc định trong run_class_finetuning.py (layer scale 0.1, rel pos bias, mean pooling).
        model = panderm_base_patch16_224_finetune(
            pretrained=False, num_classes=nb_classes, drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0,
            drop_block_rate=None, use_mean_pooling=True, init_scale=0.001, use_rel_pos_bias=True,
            init_values=0.1, lin_probe=False,
        )
        if finetuned_ckpt is not None:
            model.load_state_dict(torch.load(finetuned_ckpt, map_location="cpu")["model"], strict=True)
        elif pretrained_path is not None:
            mapped = map_pretrained_cls_keys(torch.load(pretrained_path, map_location="cpu"), model.get_num_layers())
            own = model.state_dict()
            mapped = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
            block_keys = [k for k in own if k.startswith("blocks.") and "relative_position_index" not in k]
            coverage = sum(k in mapped for k in block_keys) / max(len(block_keys), 1)
            logger.warning("Pretrained → %d/%d key khớp; độ phủ blocks %.1f%%", len(mapped), len(own), 100 * coverage)
            if coverage < 0.9:
                raise RuntimeError(f"Chỉ {coverage:.1%} trọng số blocks khớp với {pretrained_path}")
            model.load_state_dict(mapped, strict=False)
        else:
            raise ValueError("Cần finetuned_ckpt hoặc pretrained_path")
        return cls(model, device)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self.model(cls_preprocess(rgb).to(self.device))
        return torch.softmax(logits, dim=1)[0].cpu().numpy()