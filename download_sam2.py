"""Download official Meta SAM 2.1 tiny checkpoint (~150 MB)."""
from pathlib import Path
from urllib.request import urlretrieve

from sam2_core import CHECKPOINT

URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
if __name__ == "__main__":
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists() and CHECKPOINT.stat().st_size > 100_000_000:
        print("Already downloaded:", CHECKPOINT)
    else:
        tmp = CHECKPOINT.with_suffix(".download")
        try:
            print("Downloading official SAM 2.1 tiny checkpoint...")
            urlretrieve(URL, tmp)
            if tmp.stat().st_size < 100_000_000:
                raise RuntimeError("Checkpoint unexpectedly small")
            tmp.replace(CHECKPOINT)
            print("Saved:", CHECKPOINT)
        finally:
            if tmp.exists():
                tmp.unlink()
