#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


def _urls_from_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [token.strip() for token in re.split(r"[\s,]+", raw) if token.strip()]


def _direct_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.path.rstrip("/").endswith("/download"):
        return url
    if "/s/" in f"/{parsed.path.strip('/')}/":
        return url.rstrip("/") + "/download"
    return url


def _filename(response, index: int) -> str:
    header = response.headers.get("Content-Disposition", "")
    match = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", header, flags=re.IGNORECASE)
    if match:
        return Path(urllib.parse.unquote(match.group(1))).name
    match = re.search(r'filename\s*=\s*"?([^";]+)"?', header, flags=re.IGNORECASE)
    if match:
        return Path(urllib.parse.unquote(match.group(1))).name
    fallback = Path(urllib.parse.urlparse(response.geturl()).path).name
    if fallback and fallback.lower() != "download":
        return fallback
    return f"downloaded_meg_file_{index:04d}.mat"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--env-name", default="MEG_DATA_URL_LIST")
    parser.add_argument("--manifest", default="data-manifest/downloaded-files.txt")
    args = parser.parse_args()

    urls = _urls_from_env(args.env_name)
    if not urls:
        raise SystemExit(f"{args.env_name} is empty")

    data_dir = Path(args.data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for index, url in enumerate(urls, start=1):
        request = urllib.request.Request(_direct_url(url), headers={"User-Agent": "PyMEGDec"})
        with urllib.request.urlopen(request, timeout=180) as response:
            target = data_dir / _filename(response, index)
            counter = 2
            while target.exists():
                target = data_dir / f"{target.stem}_{counter}{target.suffix}"
                counter += 1
            with target.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        downloaded.append(target)
        print(f"Downloaded file #{index}: {target.name}")

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in downloaded) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
