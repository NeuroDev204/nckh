# file: scripts/prepare_isic2018.py
"""Tải + giải nén ISIC 2018 Task 1 về data_root theo đúng layout loader PanDerm, đếm số mask mỗi ảnh.

Ví dụ (.venv):  python scripts/prepare_isic2018.py --zips-dir ~/nckh_data/zips --data-root ~/nckh_data --delete-zips
"""
import argparse
import json
import logging
from pathlib import Path

from nckh.isic import ISIC2018_URLS, count_masks, download_missing, prepare_isic2018


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zips-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--skip-download", action="store_true", help="zip đã có sẵn trong --zips-dir")
    ap.add_argument("--delete-zips", action="store_true", help="xoá zip sau khi giải nén để giải phóng đĩa")
    args = ap.parse_args(argv)

    if not args.skip_download:
        download_missing(args.zips_dir, ISIC2018_URLS)
    report = prepare_isic2018(args.zips_dir, args.data_root)
    counts = count_masks(args.data_root)
    out = args.data_root / "ISIC2018" / "mask_count.json"
    out.write_text(json.dumps({"extract": report, "counts": counts}, indent=2))
    print(json.dumps(counts, indent=2))
    if args.delete_zips:
        for url in ISIC2018_URLS:
            (args.zips_dir / url.rsplit("/", 1)[-1]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
