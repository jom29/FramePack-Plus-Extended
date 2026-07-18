import os
import subprocess
from pathlib import Path
import platform
import shutil


class VideoStitcher:

    def merge(self):

        print("===================================")
        print("FramePack Video Stitcher")
        print("===================================")
        print()

        # ------------------------------------------------------------
        # Internal Project Paths
        # ------------------------------------------------------------


        if platform.system() == "Windows":

         ffmpeg_exe = Path("projects/demo/ffmpeg/ffmpeg.exe")

        else:

         ffmpeg_exe = shutil.which("ffmpeg")


        clips_folder = Path("projects/demo/segments")

        output_folder = Path("projects/demo/output")

        output_name = "final.mp4"

        # ------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------

        if ffmpeg_exe is None:

          print("\nERROR: FFmpeg not found.")

          return False

        if isinstance(ffmpeg_exe, Path):

         if not ffmpeg_exe.is_file():
 
            print("\nERROR: FFmpeg executable not found.")

            print(ffmpeg_exe)

            return False



        if not clips_folder.is_dir():
            print("\nERROR: Segment folder not found.")
            print(clips_folder)
            return False

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # ------------------------------------------------------------
        # Scan all MP4 clips (alphabetically)
        # ------------------------------------------------------------

        videos = sorted(
            f.name
            for f in clips_folder.iterdir()
            if f.suffix.lower() == ".mp4"
        )

        if not videos:
            print("\nERROR: No MP4 clips found.")
            return False

        print(f"Found {len(videos)} clips:\n")

        for video in videos:
            print(" ", video)

        # ------------------------------------------------------------
        # Create segments.txt
        # ------------------------------------------------------------

        concat_file = clips_folder / "segments.txt"

        with open(concat_file, "w", encoding="utf-8") as file:

            for video in videos:

                file.write(f"file '{video}'\n")

        # ------------------------------------------------------------
        # Output
        # ------------------------------------------------------------

        output_video = output_folder / output_name

        output_video = output_video.resolve()

        cmd = [

            str(ffmpeg_exe),

            "-y",

            "-f", "concat",

            "-safe", "0",

            "-i", "segments.txt",

            "-c", "copy",

            str(output_video)

        ]

        print("\nRunning FFmpeg...\n")


        print("--------------------------------")
        print("Current Working Directory:")
        print(clips_folder)

        print()

        print("Concat File:")
        print(concat_file)

        print()

        print("Exists:")
        print(concat_file.exists())
        print("--------------------------------")


        result = subprocess.run(

            cmd,

            cwd=str(clips_folder),

            capture_output=True,

            text=True

        )

        # ------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------

        try:
            concat_file.unlink()
        except OSError:
            pass

        # ------------------------------------------------------------
        # Finished
        # ------------------------------------------------------------

        if result.returncode != 0:

            print(result.stdout)

            print(result.stderr)

            print("Merge failed.")

            return False

        print()

        print("===================================")

        print("MERGE COMPLETED")

        print("===================================")

        print(output_video)

        return True