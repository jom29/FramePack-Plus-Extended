from pathlib import Path


# ------------------------------------------------------------
# Phase
# ------------------------------------------------------------

class Phase:

    def __init__(self, index: int):

        self.index = index

        self.name = f"Phase {index}"

        # ----------------------------------------------------
        # Keyframes
        # ----------------------------------------------------

        self.start_image = ""

        self.end_image = ""

        # ----------------------------------------------------
        # Reserved for future per-phase settings
        # ----------------------------------------------------

        self.positive_prompt = ""

        self.negative_prompt = ""

        self.duration = 5.0

        self.steps = 25

        self.resolution = 720


# ------------------------------------------------------------
# Controller
# ------------------------------------------------------------

class Controller:

    def __init__(self):

        # ----------------------------------------------------
        # Project Image Library
        # ----------------------------------------------------

        self.image_folder = Path(
            "projects/demo/images"
        )

        # ----------------------------------------------------
        # Gallery Selection
        # ----------------------------------------------------

        self.selected_image = ""

        # ----------------------------------------------------
        # Phase System
        # ----------------------------------------------------

        self.phase_list = []

        self.current_phase = 0

        # ----------------------------------------------------
        # Global Render Settings
        # ----------------------------------------------------

        self.positive_prompt = ""

        self.negative_prompt = ""

        self.duration = 5.0

        self.steps = 25

        self.resolution = 720

        self.create_default_project()

       # ----------------------------------------------------
       # Global Render Settings
       # ----------------------------------------------------

        self.global_positive_prompt = "Consistent body motion toward end keyframe"

        self.global_negative_prompt = "teleport"

        self.global_duration = 1.0

        self.global_steps = 3

        self.global_resolution = 720

        self.segment_folder = Path(
        "projects/demo/segments"
    )

    # --------------------------------------------------------
    # Project Initialization
    # --------------------------------------------------------

    def create_default_project(self):

        self.phase_list.clear()

        self.phase_list.append(
            Phase(1)
        )

        self.current_phase = 0

    # --------------------------------------------------------
    # Image Library
    # --------------------------------------------------------

    def get_images(self):

        supported = (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp"
        )

        images = []

        if not self.image_folder.exists():

            return images

        for file in sorted(self.image_folder.iterdir()):

            if file.suffix.lower() in supported:

                images.append(str(file))

        return images

    # --------------------------------------------------------
    # Gallery Selection
    # --------------------------------------------------------

    def select_image(self, image_path):

      self.selected_image = Path(image_path).name

    def get_selected_image(self):

        return self.selected_image

    




    # --------------------------------------------------------
    # Image Path Resolver
    # --------------------------------------------------------

    def resolve_image(self, filename):

     if not filename:
        return None

     return str(self.image_folder / filename)





    # --------------------------------------------------------
    # Phase Navigation
    # --------------------------------------------------------

    def get_phase_names(self):

        return [
            phase.name
            for phase in self.phase_list
        ]

    def get_current_phase(self):

        return self.phase_list[
            self.current_phase
        ]

    def select_phase(self, index):

        if 0 <= index < len(self.phase_list):

            self.current_phase = index

    # --------------------------------------------------------
    # Start / End Keyframes
    # --------------------------------------------------------

    def set_start_image(self):

        self.get_current_phase().start_image = self.selected_image

    def set_end_image(self):

        self.get_current_phase().end_image = self.selected_image

    def get_start_image(self):

     filename = self.get_current_phase().start_image

     return self.resolve_image(filename)


    def get_end_image(self):

     filename = self.get_current_phase().end_image

     return self.resolve_image(filename)





    # --------------------------------------------------------
    # Phase Management
    # --------------------------------------------------------

    def add_phase(self):

        new_index = len(self.phase_list) + 1

        self.phase_list.append(
            Phase(new_index)
        )

        self.current_phase = len(self.phase_list) - 1

    def delete_current_phase(self):

        if len(self.phase_list) <= 1:

            return

        self.phase_list.pop(
            self.current_phase
        )

        # Renumber phases

        for index, phase in enumerate(

            self.phase_list,

            start=1

        ):

            phase.index = index

            phase.name = f"Phase {index}"

        # Clamp current selection

        if self.current_phase >= len(self.phase_list):

            self.current_phase = len(self.phase_list) - 1




    # --------------------------------------------------------
    # Global Render Settings
    # --------------------------------------------------------

    def set_positive_prompt(self, value):

        self.positive_prompt = value


    def get_positive_prompt(self):

        return self.positive_prompt


    def set_negative_prompt(self, value):

        self.negative_prompt = value


    def get_negative_prompt(self):

        return self.negative_prompt


    def set_duration(self, value):

        self.duration = float(value)


    def get_duration(self):

        return self.duration


    def set_steps(self, value):

        self.steps = int(value)


    def get_steps(self):

        return self.steps


    def set_resolution(self, value):

        self.resolution = int(value)


    def get_resolution(self):

        return self.resolution


 
    # --------------------------------------------------------
    # Global Render Settings
    # --------------------------------------------------------

    def get_positive_prompt(self):

     return self.global_positive_prompt


    def get_negative_prompt(self):

     return self.global_negative_prompt


    def get_duration(self):

     return self.global_duration


    def get_steps(self):

     return self.global_steps


    def get_resolution(self):

     return self.global_resolution


    def get_segment_folder(self):

     return self.segment_folder


    # --------------------------------------------------------
    # Convenience
    # --------------------------------------------------------

    def get_phase_summary(self):

        phase = self.get_current_phase()

        return {

            "name": phase.name,

            "start_image": phase.start_image,

            "end_image": phase.end_image,

            "positive_prompt": self.positive_prompt,

            "negative_prompt": self.negative_prompt,

            "duration": self.duration,

            "steps": self.steps,

            "resolution": self.resolution,

        }