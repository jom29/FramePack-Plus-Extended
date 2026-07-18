from pathlib import Path
from project_serializer import ProjectSerializer

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
        # Settings Mode
        # ----------------------------------------------------

        self.settings_mode = "global"

        # ----------------------------------------------------
        # Global Render Settings
        # ----------------------------------------------------

        self.create_default_project()

       # ----------------------------------------------------
       # Global Render Settings
       # ----------------------------------------------------

        self.global_positive_prompt = "Consistent body motion toward end keyframe"

        self.global_negative_prompt = "teleport"

        self.global_duration = 1.0

        self.global_steps = 10

        self.global_resolution = 720


        # ----------------------------------------------------
        # Project Serializer
        # ----------------------------------------------------

        self.serializer = ProjectSerializer()

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

     self.save_project()



    def set_end_image(self):

     self.get_current_phase().end_image = self.selected_image

     self.save_project()



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

        self.global_positive_prompt = value


    def get_positive_prompt(self):

     return self.global_positive_prompt


    def set_negative_prompt(self, value):

        self.global_negative_prompt = value



    def set_duration(self, value):

        self.global_duration = float(value)




    def set_steps(self, value):

        self.global_steps = int(value)



    def set_resolution(self, value):

        self.global_resolution = int(value)


 
    # --------------------------------------------------------
    # Global Render Settings
    # --------------------------------------------------------

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
    # Project
    # --------------------------------------------------------

    def get_serializer(self):

     return self.serializer


    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------

    def save_project(self):

     self.serializer.save(self)

     print()

     print("--------------------------------")

     print("Project Saved")

     print(self.serializer.get_project_path())

     print("--------------------------------")




     # --------------------------------------------------------
     # Project
     # --------------------------------------------------------

    def load_project(self):

     loaded = self.serializer.load(self)

     if loaded:

        print()

        print("--------------------------------")

        print("Project Loaded")

        print(self.serializer.get_project_path())

        print("--------------------------------")

     else:

        print()

        print("--------------------------------")

        print("No Project Found")

        print("--------------------------------")




   # --------------------------------------------------------
   # Project Startup
   # --------------------------------------------------------

    def initialize_project(self):

     if self.serializer.exists():

        print()

        print("--------------------------------")

        print("Existing project found")

        print("--------------------------------")

        self.load_project()

     else:

        print()

        print("--------------------------------")

        print("Creating new project")

        print("--------------------------------")

        self.create_default_project()

        self.save_project()



    # --------------------------------------------------------
    # Settings Mode
    # --------------------------------------------------------

    def set_settings_mode(self, mode):

      self.settings_mode = mode

      print()

      print("--------------------------------")

      print("Settings Mode :", self.settings_mode)

      print()

      print("Inspector Values")

      print(self.load_inspector())

      print("--------------------------------")


    def get_settings_mode(self):

     return self.settings_mode



  # --------------------------------------------------------
  # Inspector
  # --------------------------------------------------------

    def load_inspector(self):

     if self.settings_mode == "global":

        return (



            self.global_positive_prompt,

            self.global_negative_prompt,

            self.global_duration,

            self.global_steps,

            self.global_resolution

        )

     phase = self.get_current_phase()

     return (

        phase.positive_prompt,

        phase.negative_prompt,

        phase.duration,

        phase.steps,

        phase.resolution

    )



    # --------------------------------------------------------
    # Inspector 
    # --------------------------------------------------------

    def commit_inspector(

      self,

     positive,

     negative,

     duration,

     steps,

     resolution

    ):

     if self.settings_mode == "global":

        self.global_positive_prompt = positive

        self.global_negative_prompt = negative

        self.global_duration = float(duration)

        self.global_steps = int(steps)

        self.global_resolution = int(resolution)

     else:

        phase = self.get_current_phase()

        phase.positive_prompt = positive

        phase.negative_prompt = negative

        phase.duration = float(duration)

        phase.steps = int(steps)

        phase.resolution = int(resolution)

     self.save_project()

    print()

    print("--------------------------------")

    print("Project Saved")

    print("--------------------------------")






    # --------------------------------------------------------
    # Settings Mode
    # --------------------------------------------------------

    def set_settings_mode(self, mode):

       self.settings_mode = mode

       print()

       print("--------------------------------")

       print("Inspector Values")

       print(self.load_inspector())

       print("--------------------------------")

    def get_settings_mode(self):

       return self.settings_mode



    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def get_render_settings(self, phase):

     if self.settings_mode == "global":

        return {

            "start_image": phase.start_image,

            "end_image": phase.end_image,

            "positive_prompt": self.global_positive_prompt,

            "negative_prompt": self.global_negative_prompt,

            "duration": self.global_duration,

            "steps": self.global_steps,

            "resolution": self.global_resolution

        }

     return {

        "start_image": phase.start_image,

        "end_image": phase.end_image,

        "positive_prompt": phase.positive_prompt,

        "negative_prompt": phase.negative_prompt,

        "duration": phase.duration,

        "steps": phase.steps,

        "resolution": phase.resolution

    }




    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def get_render_queue(self):

      queue = []

      for phase in self.phase_list:

        queue.append(

            {

               "phase": phase,

               "settings": self.get_render_settings(

              phase

                )

            }

           )

      return queue




    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def print_render_queue(self):

      queue = self.get_render_queue()

      print()

      print("========================================")
      print(" Render Queue")
      print("========================================")

      for index, item in enumerate(queue, start=1):

        settings = item["settings"]

        print()

        print(f"Phase {index}")

        print("----------------------------------------")

        print("Positive Prompt :",
              settings["positive_prompt"])

        print("Negative Prompt :",
              settings["negative_prompt"])

        print("Duration        :",
              settings["duration"])

        print("Steps           :",
              settings["steps"])

        print("Resolution      :",
              settings["resolution"])

      print()

      print("========================================")




    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    def get_phase_video(self, phase):

     video = self.segment_folder / f"phase_{phase.index:03}.mp4"

     print()
     print("--------------------------------")
     print("Current Phase :", phase.name)
     print("Video Path    :", video)
     print("Exists        :", video.exists())
     print("--------------------------------")

     if video.exists():

        return str(video)

     return None




    def get_current_phase_video(self):

     return self.get_phase_video(

        self.get_current_phase()

    )


    def get_phase_video_status(self, phase):

     if self.phase_has_video(phase):

        return "Render Complete"

     return "Not Rendered Yet"


    def get_current_phase_video_status(self):

     return self.get_phase_video_status(

        self.get_current_phase()

    )


    def phase_has_video(self, phase):

     return self.get_phase_video(

        phase

    ) is not None






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