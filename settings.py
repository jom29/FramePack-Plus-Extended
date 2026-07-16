import json
from pathlib import Path

FRAMEPACK_RUNTIME = r"E:\AI\FramePack_Official\framepack_cu126_torch26"

FRAMEPACK_WEBUI = r"E:\AI\FramePack Plus\webui"

DEFAULT_RESOLUTION = 512

DEFAULT_DURATION = 1.0

DEFAULT_STEPS = 5

DEFAULT_POSITIVE_PROMPT = (
    "Smooth continuous interpolation between the provided start and end keyframes. "
    "Preserve facial identity, body proportions, appearance, clothing, hairstyle, "
    "lighting and background."
)

DEFAULT_NEGATIVE_PROMPT = (
    "Teleportation, jump frames, flickering, identity drift, "
    "body distortion, camera jump, scene change."
)





# ------------------------------------------------------------
# Configuration File
# ------------------------------------------------------------

CONFIG_FILE = Path("config.json")


def load_paths():

    if CONFIG_FILE.exists():

        with open(CONFIG_FILE, "r", encoding="utf8") as f:

            data = json.load(f)

        runtime = data.get(
            "runtime_root",
            FRAMEPACK_RUNTIME
        )

        webui = data.get(
            "webui_root",
            FRAMEPACK_WEBUI
        )

    else:

        runtime = FRAMEPACK_RUNTIME

        webui = FRAMEPACK_WEBUI

    return runtime, webui


def save_paths(runtime, webui):

    with open(CONFIG_FILE, "w", encoding="utf8") as f:

        json.dump(

            {

                "runtime_root": runtime,

                "webui_root": webui

            },

            f,

            indent=4

        )