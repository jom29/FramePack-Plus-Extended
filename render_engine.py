from pathlib import Path
from typing import Callable
from adapter import FramePackAdapter

from settings import load_paths


class RenderEngine:

    def __init__(self, controller):

        self.controller = controller

        self.model_loaded = False

        self.cancel_requested = False



        # Existing FramePack bridge

        self.adapter = None



    def initialize_adapter(self):

        if self.adapter is not None:
          return

        print("Initializing FramePack Adapter...")





        from settings import load_paths


        runtime_root, webui_root = load_paths()

        runtime_root = Path(runtime_root)

        webui_root = Path(webui_root)
        self.adapter = FramePackAdapter(







        runtime_root=runtime_root,

        webui_root=webui_root

        )

        print("Adapter created.")


        print("Launching FramePack...")

        self.adapter.launch()

        print("Waiting for server...")

        self.adapter.wait_until_ready()

        print("Connecting client...")

        self.adapter.connect()

        print("FramePack Ready.")




    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    def load_model(self):

        """
        TODO

        Load FramePack / Hunyuan model here.

        Called only once before rendering all phases.
        """

        if self.model_loaded:
            return

        print("Loading FramePack model...")

        self.model_loaded = True

    def unload_model(self):

        """
        Optional.

        Release GPU memory after rendering.
        """

        if not self.model_loaded:
            return

        print("Unloading model...")

        self.model_loaded = False

    # --------------------------------------------------------
    # Batch Render
    # --------------------------------------------------------

    def render_project(self, progress_callback: Callable = None):

        """
        Render every phase in the project.

        progress_callback(percent, message)
        """

        self.initialize_adapter()

        self.cancel_requested = False

        self.load_model()

        phases = self.controller.phase_list

        total = len(phases)

        output_folder = Path("projects/demo/segments")

        output_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        for index, phase in enumerate(phases):

            if self.cancel_requested:

                break

            percent = int(index / total * 100)

            self._progress(

                progress_callback,

                percent,

                f"Rendering {phase.name}"

            )

            output_video = output_folder / f"phase_{index+1:03}.mp4"

            self.render_phase(

                phase,

                output_video

            )

        self._progress(

            progress_callback,

            100,

            "Finished"

        )

        self.unload_model()

    # --------------------------------------------------------
    # Single Phase
    # --------------------------------------------------------

    def render_phase(

      self,

      phase,

      output_video

    ):

       start_image = self.controller.resolve_image(
       phase.start_image
      )

       end_image = self.controller.resolve_image(
         phase.end_image
         )

       settings = self.controller.get_render_settings(
        phase
       )

       segment_folder = str(output_video.parent)

       output_name = output_video.name

       print("--------------------------------")

       print("Submitting render to FramePack...")

       print("--------------------------------")

       result = self.adapter.render_phase(
       
        start_image=start_image,

        end_image=end_image,


        prompt=settings["positive_prompt"],

        negative_prompt=settings["negative_prompt"],

        duration=settings["duration"],

        resolution=settings["resolution"],

        steps=settings["steps"],

        segment_folder=segment_folder,

        output_name=output_name

      )

       print("--------------------------------")

       print("Render Finished")

       print(result)

       print("--------------------------------")

       return result








    def _progress(

        self,

        callback,

        percent,

        message

    ):

        print(f"[{percent}%] {message}")

        if callback is not None:

            callback(

                percent,

                message

            )

    # --------------------------------------------------------
    # Cancel
    # --------------------------------------------------------

    def cancel(self):

        self.cancel_requested = True