from dataclasses import dataclass


@dataclass
class RuntimeEnvironment:

    runtime_root: str = ""

    python_executable: str = ""

    demo_gradio: str = ""

    working_directory: str = ""

    valid: bool = False