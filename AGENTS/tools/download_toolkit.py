#!/usr/bin/env python3
"""Download portable toolkit binaries for Linux or Windows."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

# --- END HEADER ---

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN_DIR = os.path.join(REPO_ROOT, "AGENTS", "tools", "bin")


class DownloadError(Exception):
    pass


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def get_asset_url(repo: str, pattern: str) -> str:
    """Return download url matching pattern from repo's latest release."""
    data = fetch_json(f"https://api.github.com/repos/{repo}/releases/latest")
    for asset in data.get("assets", []):
        if re.search(pattern, asset["name"]):
            return asset["browser_download_url"]
    raise DownloadError(f"asset not found for {repo} {pattern}")


def download_file(url: str, dest: str) -> None:
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def install_from_archive(url: str, member_match: str, dest: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, os.path.basename(url))
        download_file(url, archive)
        if archive.endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as tar:
                member = next(m for m in tar.getmembers() if re.search(member_match, m.name))
                member.name = os.path.basename(member.name)
                tar.extract(member, tmp)
                shutil.move(os.path.join(tmp, member.name), dest)
        elif archive.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                member = next(m for m in zf.namelist() if re.search(member_match, m))
                zf.extract(member, tmp)
                shutil.move(os.path.join(tmp, member), dest)
        else:
            raise DownloadError(f"Unknown archive type: {archive}")
    os.chmod(dest, 0o755)


def fetch_linux() -> None:
    os.makedirs(BIN_DIR, exist_ok=True)
    download_file(
        "https://busybox.net/downloads/binaries/1.36.0-defconfig-multiarch/busybox-x86_64",
        os.path.join(BIN_DIR, "busybox"),
    )
    os.chmod(os.path.join(BIN_DIR, "busybox"), 0o755)
    download_file("https://musl.cc/x86_64-linux-musl/nano", os.path.join(BIN_DIR, "nano"))
    os.chmod(os.path.join(BIN_DIR, "nano"), 0o755)
    install_from_archive(
        get_asset_url("sharkdp/bat", r"x86_64-unknown-linux-musl\.tar\.gz"),
        r"/bat$",
        os.path.join(BIN_DIR, "bat"),
    )
    install_from_archive(
        get_asset_url("junegunn/fzf", r"linux_amd64\.tar\.gz"),
        r"/fzf$",
        os.path.join(BIN_DIR, "fzf"),
    )
    install_from_archive(
        get_asset_url("BurntSushi/ripgrep", r"x86_64-unknown-linux-musl\.tar\.gz"),
        r"/rg$",
        os.path.join(BIN_DIR, "rg"),
    )


def fetch_windows() -> None:
    os.makedirs(BIN_DIR, exist_ok=True)
    download_file(
        "https://frippery.org/files/busybox/busybox.exe",
        os.path.join(BIN_DIR, "busybox.exe"),
    )
    install_from_archive(
        get_asset_url("sharkdp/bat", r"x86_64-pc-windows-msvc\.zip"),
        r"bat.exe$",
        os.path.join(BIN_DIR, "bat.exe"),
    )
    install_from_archive(
        get_asset_url("junegunn/fzf", r"windows_amd64\.zip"),
        r"fzf.exe$",
        os.path.join(BIN_DIR, "fzf.exe"),
    )
    install_from_archive(
        get_asset_url("BurntSushi/ripgrep", r"x86_64-pc-windows-msvc\.zip"),
        r"rg.exe$",
        os.path.join(BIN_DIR, "rg.exe"),
    )
    # Windows nano builds are less consistent; attempt one known source.
    try:
        install_from_archive(
            "https://github.com/lhmouse/nano-win/releases/download/v8.4/nano-win_8.4.zip",
            r"nano.exe$",
            os.path.join(BIN_DIR, "nano.exe"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"nano.exe unavailable: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch agent toolkit binaries")
    parser.add_argument(
        "--target",
        choices=["linux", "windows"],
        default="linux" if platform.system() != "Windows" else "windows",
    )
    args = parser.parse_args()
    if args.target == "linux":
        fetch_linux()
    else:
        fetch_windows()
    print("Toolkit download complete. Binaries saved to", BIN_DIR)


if __name__ == "__main__":
    main()
