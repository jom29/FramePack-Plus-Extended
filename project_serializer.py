import json
from pathlib import Path


class ProjectSerializer:

    PROJECT_VERSION = 1

    def __init__(self, project_folder="projects/demo"):

        self.project_folder = Path(project_folder)

        self.userpref_folder = self.project_folder / "userpref_data"

        self.project_file = self.userpref_folder / "project.json"

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    def ensure_directory(self):

        self.userpref_folder.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save(self, controller):

        self.ensure_directory()

        data = {

            "project_version": self.PROJECT_VERSION,

            "project_name": self.project_folder.name,

            "settings_mode": controller.settings_mode,

            "current_phase": controller.current_phase,

            "global_settings": {

                "positive_prompt":
                    controller.global_positive_prompt,

                "negative_prompt":
                    controller.global_negative_prompt,

                "duration":
                    controller.global_duration,

                "steps":
                    controller.global_steps,

                "resolution":
                    controller.global_resolution
            },

            "phases": []

        }

        for phase in controller.phase_list:

            phase_data = {

                "id":
                    phase.index,

                "name":
                    phase.name,

                "start_image":
                    phase.start_image,

                "end_image":
                    phase.end_image,

                "local_settings": {

                    "positive_prompt":
                        phase.positive_prompt,

                    "negative_prompt":
                        phase.negative_prompt,

                    "duration":
                        phase.duration,

                    "steps":
                        phase.steps,

                    "resolution":
                        phase.resolution
                }

            }

            data["phases"].append(
                phase_data
            )

        with open(

            self.project_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                data,

                file,

                indent=4,

                ensure_ascii=False

            )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    def load(self, controller):

        self.ensure_directory()

        if not self.project_file.exists():

            return False

        with open(

            self.project_file,

            "r",

            encoding="utf-8"

        ) as file:

            data = json.load(file)

        controller.settings_mode = data.get(

            "settings_mode",

            "global"

        )

        controller.current_phase = data.get(

            "current_phase",

            0

        )

        global_settings = data.get(

            "global_settings",

            {}

        )

        controller.global_positive_prompt = global_settings.get(

            "positive_prompt",

            ""

        )

        controller.global_negative_prompt = global_settings.get(

            "negative_prompt",

            ""

        )

        controller.global_duration = global_settings.get(

            "duration",

            1.0

        )

        controller.global_steps = global_settings.get(

            "steps",

            10

        )

        controller.global_resolution = global_settings.get(

            "resolution",

            720

        )

        controller.phase_list.clear()

        from controller import Phase

        for item in data.get("phases", []):

            phase = Phase(

                item.get("id", 1)

            )

            phase.name = item.get(

                "name",

                f"Phase {phase.index}"

            )

            phase.start_image = item.get(

                "start_image",

                ""

            )

            phase.end_image = item.get(

                "end_image",

                ""

            )

            local = item.get(

                "local_settings",

                {}

            )

            phase.positive_prompt = local.get(

                "positive_prompt",

                ""

            )

            phase.negative_prompt = local.get(

                "negative_prompt",

                ""

            )

            phase.duration = local.get(

                "duration",

                5.0

            )

            phase.steps = local.get(

                "steps",

                25

            )

            phase.resolution = local.get(

                "resolution",

                720

            )

            controller.phase_list.append(

                phase

            )

        if len(controller.phase_list) == 0:

            controller.create_default_project()

        if controller.current_phase >= len(controller.phase_list):

            controller.current_phase = 0

        return True

    # --------------------------------------------------------
    # Create New Project
    # --------------------------------------------------------

    def create_default(self, controller):

        controller.create_default_project()

        self.save(controller)

    # --------------------------------------------------------
    # Delete Project File
    # --------------------------------------------------------

    def delete_project_file(self):

        if self.project_file.exists():

            self.project_file.unlink()

    # --------------------------------------------------------
    # Exists
    # --------------------------------------------------------

    def exists(self):

        return self.project_file.exists()

    # --------------------------------------------------------
    # Get Path
    # --------------------------------------------------------

    def get_project_path(self):

        return self.project_file