
import os
import subprocess


def main():

    print("===================================")
    print("FramePack Video Stitcher")
    print("===================================")
    print()

    ffmpeg_exe = input("FFmpeg Executable : ").strip()

    if not os.path.isfile(ffmpeg_exe):
        print("\nERROR: FFmpeg executable not found.")
        return

    clips_folder = input("Target Clips Folder : ").strip()

    if not os.path.isdir(clips_folder):
        print("\nERROR: Clips folder not found.")
        return

    output_folder = input("Output Folder : ").strip()

    os.makedirs(output_folder, exist_ok=True)

    output_name = input("Output Video Name [final.mp4] : ").strip()

    if not output_name:
        output_name = "final.mp4"

    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

    # ------------------------------------------------------------
    # Scan all MP4 clips (alphabetically)
    # ------------------------------------------------------------

    videos = sorted(
        f for f in os.listdir(clips_folder)
        if f.lower().endswith(".mp4")
    )

    if not videos:
        print("\nERROR: No MP4 files found in the selected folder.")
        return

    print()
    print(f"Found {len(videos)} clips:\n")

    for v in videos:
        print(" ", v)

    answer = input("\nProceed with merge? (Y/N): ").strip().lower()

    if answer != "y":
        print("Cancelled.")
        return

    concat_file = os.path.join(clips_folder, "segments.txt")

    with open(concat_file, "w", encoding="utf-8") as f:
        for v in videos:
            f.write(f"file '{v}'\n")

    output_video = os.path.join(output_folder, output_name)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_video,
    ]

    print("\nRunning FFmpeg...\n")

    result = subprocess.run(
        cmd,
        cwd=clips_folder,
        capture_output=True,
        text=True
    )

    try:
        os.remove(concat_file)
    except OSError:
        pass

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print("Merge failed.")
        return

    print("===================================")
    print("MERGE COMPLETED")
    print("===================================")
    print(output_video)


if __name__ == "__main__":
    main()
