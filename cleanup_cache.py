#!/usr/bin/env python3

import os
import shutil
from pathlib import Path


# ============================================================
# PROTECTED PATHS
# ============================================================

# IMPORTANT:
# The entire FramePackPlus folder is protected.
# This automatically protects:
#
# /workspace/FramePackPlus/hf_download
# /workspace/FramePackPlus/loras
# /workspace/FramePackPlus/outputs
# /workspace/FramePackPlus/demo_gradio.py
# and everything else inside FramePackPlus.
#
# The entire FramePack-Plus-Extended project is also protected.

PROTECTED_PATHS = [
    Path("/workspace/FramePackPlus"),
    Path("/workspace/FramePack-Plus-Extended"),
]


# ============================================================
# CACHE PATHS
# ============================================================

CACHE_PATHS = [
    Path("/tmp"),
    Path("/root/.cache"),
    Path("/root/.nv/ComputeCache"),
    Path("/var/cache/apt/archives"),
]


# ============================================================
# HELPERS
# ============================================================

def is_protected(path: Path) -> bool:
    """Return True if path is inside a protected folder."""

    try:
        path = path.resolve()

        for protected in PROTECTED_PATHS:
            protected = protected.resolve()

            if path == protected:
                return True

            if protected in path.parents:
                return True

        return False

    except Exception:
        return False


def remove_contents(directory: Path):
    """Remove contents of a cache directory without deleting the directory itself."""

    if not directory.exists():
        print(f"[SKIP] {directory} does not exist")
        return

    if not directory.is_dir():
        print(f"[SKIP] {directory} is not a directory")
        return

    print(f"\n[CLEAN] {directory}")

    for item in directory.iterdir():

        if is_protected(item):
            print(f"  [PROTECTED] {item}")
            continue

        try:

            if item.is_symlink() or item.is_file():
                item.unlink()
                print(f"  [DELETE] {item}")

            elif item.is_dir():
                shutil.rmtree(item)
                print(f"  [DELETE DIR] {item}")

        except Exception as e:
            print(f"  [FAILED] {item}")
            print(f"           {e}")


def remove_pycache_except_protected(root: Path):
    """Remove __pycache__ directories while respecting protected paths."""

    if not root.exists():
        return

    print(f"\n[CLEAN PYTHON CACHE] {root}")

    for current_root, dirs, files in os.walk(root, topdown=True):

        current = Path(current_root)

        # Never enter protected directories.
        dirs[:] = [
            d for d in dirs
            if not is_protected(current / d)
        ]

        if "__pycache__" in dirs:

            cache_dir = current / "__pycache__"

            if is_protected(cache_dir):
                continue

            try:
                shutil.rmtree(cache_dir)
                print(f"  [DELETE] {cache_dir}")

                dirs.remove("__pycache__")

            except Exception as e:
                print(f"  [FAILED] {cache_dir}")
                print(f"           {e}")


def verify_protected():
    print("\n" + "=" * 60)
    print("VERIFYING PROTECTED PATHS")
    print("=" * 60)

    all_ok = True

    for path in PROTECTED_PATHS:

        if path.exists():
            print(f"[OK] {path}")
        else:
            print(f"[ERROR] MISSING: {path}")
            all_ok = False

    return all_ok


def show_disk_usage():

    print("\n" + "=" * 60)
    print("DISK USAGE")
    print("=" * 60)

    os.system("df -h /")
    os.system("df -h /workspace")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SAFE RUNPOD CACHE CLEANER")
    print("=" * 60)

    print("\nPROTECTED DIRECTORIES:")

    for path in PROTECTED_PATHS:
        print(f"  [PROTECTED] {path}")

    print("\nCACHE LOCATIONS TO CLEAN:")

    for path in CACHE_PATHS:
        print(f"  [CACHE] {path}")

    print("\nThe entire FramePackPlus folder is protected.")
    print("The entire FramePack-Plus-Extended folder is protected.")
    print("No files inside those folders will be deleted.")

    # --------------------------------------------------------
    # Clean standard caches
    # --------------------------------------------------------

    for cache in CACHE_PATHS:
        remove_contents(cache)

    # --------------------------------------------------------
    # Clean Python bytecode cache
    #
    # NOTE:
    # FramePackPlus is protected, so this does NOT touch it.
    # --------------------------------------------------------

    # No project cleanup is performed here.
    #
    # This is intentional because the entire
    # FramePackPlus and FramePack-Plus-Extended folders
    # are protected.

    # --------------------------------------------------------
    # Verify protected directories
    # --------------------------------------------------------

    if verify_protected():
        print("\n[SAFE] All protected directories still exist.")
    else:
        print("\n[WARNING] A protected directory is missing.")

    # --------------------------------------------------------
    # Show final disk usage
    # --------------------------------------------------------

    show_disk_usage()

    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()