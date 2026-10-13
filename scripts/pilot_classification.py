"""Two optimizer steps with real train labels, including checkpoint resume."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from nckh.infer import ClsPredictor, cls_preprocess


def run_pilot(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor,
              output: Path) -> dict:
    keys = ("patch_embed.proj.weight", "head.weight")
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05)
    reports = []
    for step in (1, 2):
        before = {key: parameters[key].detach().clone() for key in keys}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if float(gradient_norm) <= 0:
            raise RuntimeError("Zero gradient norm")
        for key in keys:
            gradient = parameters[key].grad
            if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().max() > 0:
                raise RuntimeError(f"Invalid gradient: {key}")
        optimizer.step()
        updates = {key: float((parameters[key].detach() - before[key]).abs().max()) for key in keys}
        if not all(value > 0 for value in updates.values()):
            raise RuntimeError(f"Weights did not change: {updates}")
        optimizer_step = max(int(state["step"]) for state in optimizer.state.values())
        if optimizer_step != step:
            raise RuntimeError(f"Unexpected optimizer step: {optimizer_step}")
        reports.append({"step": step, "loss": float(loss.detach()),
                        "grad_norm": float(gradient_norm), "updates": updates})
        model.eval()
        with torch.no_grad():
            expected = model(images).detach().cpu()
        checkpoint = output / f"checkpoint-{step}.pth"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, checkpoint)
        if step == 1:
            # Corrupt the head first so the reload check actually tests restoration.
            with torch.no_grad():
                parameters["head.weight"].zero_()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model"], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05)
            optimizer.load_state_dict(saved["optimizer"])
            if max(int(state["step"]) for state in optimizer.state.values()) != 1:
                raise RuntimeError("Optimizer step was not restored")
            with torch.no_grad():
                actual = model(images).detach().cpu()
            if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
                raise RuntimeError("Checkpoint logits do not match")
            reload_error = float((actual - expected).abs().max())
            del saved
    return {"status": "PASS", "synthetic_targets": False, "stages": reports,
            "restored_optimizer_step": 1, "max_abs_logits_error": reload_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panderm-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This model pilot requires a GPU")
    table = pd.read_csv(args.manifest)
    train = table[table["split"] == "train"]
    if len(train) < 2:
        raise ValueError("Need at least two train images")
    selected = train.sample(n=2, random_state=2026)
    if not selected["label"].isin([0, 1, 2]).all():
        raise ValueError("Invalid classification labels")
    torch.manual_seed(2026)
    np.random.seed(2026)
    tensors = []
    for image in selected["image"]:
        with Image.open(args.data_root / "ISIC2017" / image) as rgb:
            tensors.append(cls_preprocess(np.asarray(rgb.convert("RGB"))))
    images = torch.cat(tensors).to("cuda")
    labels = torch.tensor(selected["label"].tolist(), device="cuda", dtype=torch.long)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(args.out_dir / "selected_train.csv", index=False)
    model = ClsPredictor.from_checkpoint(args.panderm_dir, pretrained_path=args.pretrained, device="cuda").model
    torch.cuda.reset_peak_memory_stats()
    report = run_pilot(model, images, labels, args.out_dir)
    report.update(task="cls", seed=2026, n=2, split="train",
                  peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PILOT CLASSIFICATION OK — technical check, not performance evaluation")


if __name__ == "__main__":
    main()
