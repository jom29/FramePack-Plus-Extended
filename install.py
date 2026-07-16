import subprocess
import sys
from pathlib import Path


def run(command, cwd=None):

    print()
    print("=" * 60)
    print("Running:", " ".join(command))
    print("=" * 60)

    result = subprocess.run(command, cwd=cwd)

    if result.returncode != 0:

        print()
        print("Installation failed.")
        sys.exit(result.returncode)


def main():

    print()
    print("=" * 60)
    print("FramePack Plus Extended Installer")
    print("=" * 60)

    framepack_folder = input(
        "FramePack Plus Folder\n> "
    ).strip()

    while True:

        folder = Path(framepack_folder)

        if not folder.exists():

            print("Folder does not exist.")
            framepack_folder = input("> ").strip()
            continue

        if not (folder / "requirements.txt").exists():

            print("requirements.txt not found.")
            framepack_folder = input("> ").strip()
            continue

        break

    print()
    print("Installing dependencies...")
    print()

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "gradio-client"
        ]
    )

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "requests"
        ]
    )

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt"
        ],
        cwd=framepack_folder
    )

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "hf_transfer"
        ]
    )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "gradio==5.22.0"
        ]
    )

    print()
    print("=" * 60)
    print("Installation Complete")
    print("=" * 60)
    print()


if __name__ == "__main__":

    main()