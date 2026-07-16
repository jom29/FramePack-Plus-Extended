from pathlib import Path
from typing import Callable


class RenderEngine:

    def __init__(self, controller):

        self.controller = controller

        self.model_loaded = False

        self.cancel_requested = False

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

        """
        This will later contain the code from main.py.

        For now it only prints.
        """

        print("-----------------------------")

        print("Render")

        print("Start :", phase.start_image)

        print("End   :", phase.end_image)

        print("Output:", output_video)

        print("-----------------------------")

        #
        # TODO
        #
        # Replace this section with the actual
        # FramePack rendering call.
        #

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

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