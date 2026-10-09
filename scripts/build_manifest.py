# file: scripts/build_manifest.py
"""Tạo manifest ISIC 2018 (3 split chính thức), báo ảnh trùng xuyên split, ghi hash của manifest.

Ví dụ: python scripts/build_manifest.py --data-root ~/nckh_data --out ~/nckh_root/data/manifests/isic2018_seg.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from nckh.manifest import build_isic_seg_manifest, find_cross_split_duplicates
from nckh.runcard import sha256_file


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    df = pd.concat([build_isic_seg_manifest(args.data_root, s) for s in ("train", "val", "test")], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    dups = find_cross_split_duplicates(df)
    dups.to_csv(args.out.with_suffix(".cross_split_duplicates.csv"), index=False)
    digest = sha256_file(args.out)
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(f"{digest}  {args.out.name}\n")

    print(df.groupby("split").size().rename("n_images"))
    print("Lý do loại:", df["exclude_reason"].replace("", "(giữ)").value_counts().to_dict())
    print("Ảnh trùng hash xuyên split:", len(dups))
    print("SHA-256 manifest:", digest)


if __name__ == "__main__":
    main()
