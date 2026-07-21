from pathlib import Path
from motion_timeline import MotionTimeline
from multikey_adapter import MultiKeyAdapter



class MultiKeyApplication:

    def __init__(self):

        self.runtime_root = ""

        self.webui_root = ""

        self.output_folder = ""

        self.prompt = ""

        self.negative_prompt = ""

        self.duration = 5

        self.steps = 25

        self.resolution = 640

    # --------------------------------------------------

    def run(self):

        self.collect_configuration()

        self.print_summary()

        print()
        self.build_motion_timeline()

        print()

        adapter = MultiKeyAdapter()

        adapter.render_motion(

        timeline=self.timeline,

        prompt=self.prompt,

        negative_prompt=self.negative_prompt,

        duration=self.duration,

        resolution=self.resolution,

        steps=self.steps,

        runtime_root=self.runtime_root,

        webui_root=self.webui_root,

        output_folder=self.output_folder

     )

     # --------------------------------------------------


    # --------------------------------------------------

    def build_motion_timeline(self):

     self.timeline = MotionTimeline()

     print()

     print("========================================")
     print("Motion Timeline")
     print("========================================")

     count = int(

        input(

            "How many KeyFrames?\n> "

        )

     )

     print()

     for i in range(count):

        image = input(

            f"KeyFrame {i + 1}\n> "

        ).strip().replace('"', "")

        self.timeline.add_pose(

            image

        )

        print()

     self.timeline.load_images()

     self.timeline.print_summary()






    def collect_configuration(self):

        print()

        print("========================================")

        print("FramePack Multi-KeyFrame Research")

        print("========================================")

        print()

        self.runtime_root = input(

            "Official Runtime Folder\n> "

        ).strip().replace('"', "")

        print()

        self.webui_root = input(

            "FramePack Plus WebUI Folder\n> "

        ).strip().replace('"', "")

        print()

        self.output_folder = input(

            "Output Folder\n> "

        ).strip().replace('"', "")

        print()

        prompt = input(

            "Positive Prompt\n> "

        ).strip()

        if prompt:

            self.prompt = prompt

        print()

        negative = input(

            "Negative Prompt\n> "

        ).strip()

        if negative:

            self.negative_prompt = negative

        print()

        duration = input(

            "Duration [5]\n> "

        ).strip()

        if duration:

            self.duration = float(duration)

        print()

        resolution = input(

            "Resolution [640]\n> "

        ).strip()

        if resolution:

            self.resolution = int(resolution)

        print()

        steps = input(

            "Steps [25]\n> "

        ).strip()

        if steps:

            self.steps = int(steps)

    # --------------------------------------------------

    def print_summary(self):

        print()

        print("========================================")

        print("Configuration")

        print("========================================")

        print()

        print("Runtime")

        print(self.runtime_root)

        print()

        print("WebUI")

        print(self.webui_root)

        print()

        print("Output")

        print(self.output_folder)

        print()

        print("Prompt")

        print(self.prompt)

        print()

        print("Negative")

        print(self.negative_prompt)

        print()

        print("Duration")

        print(self.duration)

        print()

        print("Resolution")

        print(self.resolution)

        print()

        print("Steps")

        print(self.steps)


if __name__ == "__main__":

    MultiKeyApplication().run()