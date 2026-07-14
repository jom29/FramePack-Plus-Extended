from config import Project, Phase

from settings import *

from adapter import FramePackAdapter


def main():

    project = Project()

    # ------------------------------------------------------------
    # Runtime Configuration
    # ------------------------------------------------------------

    project.render.runtime_root = (
        input(f"FramePack Runtime [{FRAMEPACK_RUNTIME}] : ").strip()
        or FRAMEPACK_RUNTIME
    )

    project.render.webui_root = (
        input(f"FramePack Plus WebUI [{FRAMEPACK_WEBUI}] : ").strip() or FRAMEPACK_WEBUI
    )

    # ------------------------------------------------------------
    # Render Settings
    # ------------------------------------------------------------

    res = input(f"Resolution [{DEFAULT_RESOLUTION}] : ").strip()

    project.render.resolution = int(res) if res else DEFAULT_RESOLUTION

    dur = input(f"Duration [{DEFAULT_DURATION}] : ").strip()

    project.render.duration = float(dur) if dur else DEFAULT_DURATION

    steps = input(f"Steps [{DEFAULT_STEPS}] : ").strip()

    project.render.steps = int(steps) if steps else DEFAULT_STEPS

    project.render.positive_prompt = DEFAULT_POSITIVE_PROMPT
    project.render.negative_prompt = DEFAULT_NEGATIVE_PROMPT

    print("\nGlobal prompts loaded.\n")

    # ------------------------------------------------------------
    # Input Phases
    # ------------------------------------------------------------

    for i in range(3):

        print(f"\nPhase {i + 1}")

        phase = Phase()

        phase.start_image = input("Start Image : ").strip()

        phase.end_image = input("End Image : ").strip()

        phase.output_name = f"segment_{i + 1:03}.mp4"

        project.phases.append(phase)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("\n===================================")
    print("PROJECT SUMMARY")
    print("===================================")

    print(f"Runtime    : {project.render.runtime_root}")
    print(f"WebUI      : {project.render.webui_root}")
    print(f"Resolution : {project.render.resolution}")
    print(f"Duration   : {project.render.duration}")
    print(f"Steps      : {project.render.steps}")

    print()

    for index, phase in enumerate(project.phases, start=1):

        print("-----------------------------------")
        print(f"Phase {index}")
        print(f"Start  : {phase.start_image}")
        print(f"End    : {phase.end_image}")
        print(f"Output : {phase.output_name}")

    # ------------------------------------------------------------
    # Launch FramePack
    # ------------------------------------------------------------

    adapter = FramePackAdapter(
        runtime_root=project.render.runtime_root, webui_root=project.render.webui_root
    )

    try:

        print()

        print("===================================")
        print("Launching FramePack Plus")
        print("===================================")

        adapter.launch()

        adapter.wait_until_ready()

        adapter.connect()

        print()

        print("===================================")
        print("Starting Render Test")
        print("===================================")

        result = adapter.render_phase(
            start_image=project.phases[0].start_image,
            end_image=project.phases[0].end_image,
            prompt=project.render.positive_prompt,
            negative_prompt=project.render.negative_prompt,
            duration=project.render.duration,
            resolution=project.render.resolution,
            steps=project.render.steps,
        )

        print()

        print("===================================")
        print("Render Result")
        print("===================================")

        print(result)

        input("\nPress ENTER to shutdown FramePack...")

    except Exception as ex:

        print()

        print("===================================")
        print("ERROR")
        print("===================================")

        adapter.print_server_log()

        print(ex)

    finally:

        adapter.shutdown()


if __name__ == "__main__":

    main()
