from dataclasses import dataclass, field
from typing import List


@dataclass
class Phase:

    start_image: str = ""
    end_image: str = ""
    output_name: str = ""


@dataclass
class RenderSettings:

    runtime_root: str = ""

    webui_root: str = ""

    resolution: int = 512

    duration: float = 1.0

    steps: int = 5

    positive_prompt: str = ""

    negative_prompt: str = ""


@dataclass
class Project:

    render: RenderSettings = field(default_factory=RenderSettings)

    phases: List[Phase] = field(default_factory=list)