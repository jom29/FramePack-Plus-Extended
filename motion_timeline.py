from dataclasses import dataclass
from pathlib import Path
from PIL import Image


# ------------------------------------------------------------
# Motion KeyPose
# ------------------------------------------------------------

@dataclass
class MotionKeyPose:

    index: int

    image_path: Path

    image: Image.Image | None = None


# ------------------------------------------------------------
# Motion Timeline
# ------------------------------------------------------------

class MotionTimeline:

    def __init__(self):

        self.keyposes = []

    # --------------------------------------------------------

    def add_pose(self, image_path):

        pose = MotionKeyPose(

            index=len(self.keyposes),

            image_path=Path(image_path)

        )

        self.keyposes.append(pose)

    # --------------------------------------------------------

    def load_images(self):

        print()

        print("========================================")
        print("Loading KeyFrames")
        print("========================================")

        for pose in self.keyposes:

            if not pose.image_path.exists():

                raise FileNotFoundError(

                    pose.image_path

                )

            pose.image = Image.open(

                pose.image_path

            ).convert("RGB")

            print(

                f"[{pose.index + 1}]",

                pose.image_path.name,

                f"{pose.image.width}x{pose.image.height}"

            )

    # --------------------------------------------------------

    def print_summary(self):

        print()

        print("========================================")
        print("Motion Timeline")
        print("========================================")

        for pose in self.keyposes:

            print(

                f"{pose.index + 1}.",

                pose.image_path.name

            )