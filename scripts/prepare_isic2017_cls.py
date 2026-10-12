# file: scripts/prepare_isic2017_cls.py
"""Tải ảnh + nhãn ISIC 2017 Task 3 và tạo CSV cho run_class_finetuning.py của PanDerm.

Sinh hai file trong --out-dir:
  isic2017_cls.csv            split thật (train/val/test) — chỉ dùng cho lần eval cuối
  isic2017_cls_trainphase.csv dòng test = bản sao val    — dùng khi fine-tune (xem P5b)
Ví dụ (.venv):
  python scripts/prepare_isic2017_cls.py --zips-dir ~/nckh_data/zips --data-root ~/nckh_data --out-dir ~/nckh_root/data/manifests
"""
import argparse
import logging
from pathlib import Path

from nckh.isic import (
    CLASS_NAMES,
    ISIC2017_GT_CSV,
    ISIC2017_GT_URLS,
    ISIC2017_IMAGE_DIRS,
    ISIC2017_URLS,
    download_missing,
    make_cls_table,
    make_trainphase,
    prepare_isic2017_images,
)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zips-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="nơi ghi CSV (nên là NCKH_ROOT/data/manifests)")
    ap.add_argument("--labels-only", action="store_true", help="chỉ tải CSV nhãn, không tải ảnh")
    ap.add_argument("--delete-zips", action="store_true")
    args = ap.parse_args(argv)

    download_missing(args.zips_dir, ISIC2017_GT_URLS)
    if not args.labels_only:
        download_missing(args.zips_dir, ISIC2017_URLS)
        print(prepare_isic2017_images(args.zips_dir, args.data_root))

    df = make_cls_table({s: args.zips_dir / f for s, f in ISIC2017_GT_CSV.items()}, ISIC2017_IMAGE_DIRS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "isic2017_cls.csv", index=False)
    make_trainphase(df).to_csv(args.out_dir / "isic2017_cls_trainphase.csv", index=False)
    table = df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(CLASS_NAMES)))
    print(table)
    if args.delete_zips:
        for url in ISIC2017_URLS:
            (args.zips_dir / url.rsplit("/", 1)[-1]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
