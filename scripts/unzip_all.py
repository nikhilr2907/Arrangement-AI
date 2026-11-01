"""
Unzip all .zip files in a source folder into a destination folder.

Usage:
    python scripts/unzip_all.py ./data D:/train_stems
"""

import argparse
import zipfile
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unzip all .zip files in a folder")
    parser.add_argument("folder", help="Folder to scan for .zip files")
    parser.add_argument("dest",            help="Destination folder to extract into")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already exist in the destination")
    args = parser.parse_args()

    folder = Path(args.folder)
    dest   = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    zips = sorted(folder.glob("*.zip"))

    if not zips:
        print(f"No .zip files found in {folder}")
    else:
        for zip_path in zips:
            if not zipfile.is_zipfile(zip_path):
                print(f"  SKIP {zip_path.name}: not a valid zip file")
                continue
            print(f"Extracting {zip_path.name} → {dest}/")
            with zipfile.ZipFile(zip_path) as zf:
                members = [
                    m for m in zf.infolist()
                    if not args.skip_existing or not (dest / m.filename).exists()
                ]
                zf.extractall(dest, members=members)
        print(f"\nDone. {len(zips)} archive(s) extracted.")
