import os
from pathlib import Path
import gradio as gr
from urllib.parse import quote
import sys
from motion_timeline import MotionTimeline
from multikey_adapter import MultiKeyAdapter

def reverse_render_sequence(timeline):
    """
    Convert GUI timeline order into the order expected
    by the render pipeline.
    """

    return list(reversed(timeline))






def build_render_timeline(timeline, gallery_images):
    render_timeline = reverse_render_sequence(timeline)

    motion_timeline = MotionTimeline()

    for image_index in render_timeline:
        image_path = gallery_images[image_index][0]

        motion_timeline.add_pose(image_path)

    motion_timeline.load_images()

    return motion_timeline






# ============================================================
# W6.1.2 - Reverse Sequence Test
# ============================================================

print("========================================")
print("REVERSE SEQUENCE TEST")
print("========================================")

test_timeline = [0, 1, 2]

print("GUI Timeline   :", test_timeline)

render_timeline = reverse_render_sequence(test_timeline)

print("Render Timeline:", render_timeline)

print("========================================")






# ============================================================
# Runtime Path Helpers
# ============================================================

def resolve_python_executable(runtime_root):
    """
    Find the Python executable inside the configured
    FramePack runtime on Windows or Linux.
    """

    runtime_root = Path(runtime_root).expanduser()

    candidates = [
        # Windows virtual environment
        runtime_root / "venv" / "Scripts" / "python.exe",

        # Linux virtual environment
        runtime_root / "venv" / "bin" / "python",

        # Windows embedded Python
        runtime_root / "system" / "python" / "python.exe",

        # Linux/system-style Python location
        runtime_root / "system" / "python" / "bin" / "python",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Fallback to the Python running this WebGUI
    return Path(sys.executable)


def resolve_path(value):
    """
    Convert a user-entered path into a normalized Path.
    Works on Windows and Linux.
    """

    if not value:
        return None

    return Path(value).expanduser().resolve()




def validate_paths(runtime, webui, output):
    """
    Validate configured runtime, WebUI and output paths.
    """

    runtime = resolve_path(runtime)
    webui = resolve_path(webui)
    output = resolve_path(output)

    messages = []

    if runtime is None:
        messages.append("FramePack Runtime Path is empty.")
    elif not runtime.is_dir():
        messages.append(f"Runtime path does not exist: {runtime}")

    if webui is None:
        messages.append("FramePack WebUI Path is empty.")
    elif not webui.is_dir():
        messages.append(f"WebUI path does not exist: {webui}")

    if output is None:
        messages.append("Output Folder is empty.")

    if messages:
        return "\n".join(messages)

    # Output is allowed not to exist yet.
    output.mkdir(parents=True, exist_ok=True)

    python_executable = resolve_python_executable(runtime)

    return (
        "Paths OK\n\n"
        f"Runtime : {runtime}\n"
        f"WebUI   : {webui}\n"
        f"Output  : {output}\n"
        f"Python  : {python_executable}"
    )



# ============================================================
# Internal Project Paths
# ============================================================

APP_ROOT = Path(__file__).resolve().parent

PROJECTS_ROOT = APP_ROOT / "projects"

CURRENT_PROJECT = "demo"


def image_folder(project=CURRENT_PROJECT):
    """
    Return the internal image folder for the selected project.

    Cross-platform:
        Windows -> APP_ROOT\\projects\\demo\\images
        Linux   -> APP_ROOT/projects/demo/images
    """
    return PROJECTS_ROOT / project / "images"


def scan_images(project=CURRENT_PROJECT):
    """
    Scan the project's internal Image Library.
    """

    folder = image_folder(project)

    if not folder.is_dir():
        return []

    exts = (".png", ".jpg", ".jpeg", ".webp")

    return [
        str(path)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in exts
    ]


def build_render_config(
    timeline,
    positive_prompt,
    negative_prompt,
    duration,
    steps,
    resolution,
    cfg_scale,
    seed
):
    render_timeline = reverse_render_sequence(timeline)

    config = {
        "keyframes": render_timeline,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "duration": float(duration),
        "steps": int(steps),
        "resolution": resolution,
        "cfg_scale": float(cfg_scale),
        "seed": int(seed),

        # Hidden render defaults
        "latent_window_size": 9,
        "gs": 10.0,
        "rs": 0.0,
        "gpu_memory_preservation": 6,
        "use_teacache": True,
        "mp4_crf": 16,
        "teacache_threshold": 0.15,
        "lora_file": None,
        "lora_multiplier": 0.8,
        "fp8_optimization": False,
    }

    return config





def test_render_config(
    timeline,
    positive_prompt,
    negative_prompt,
    duration,
    steps,
    resolution,
    cfg_scale,
    seed
):
    config = build_render_config(
        timeline,
        positive_prompt,
        negative_prompt,
        duration,
        steps,
        resolution,
        cfg_scale,
        seed
    )

    print()
    print("========================================")
    print("W6.3.2 : RENDER CONFIGURATION TEST")
    print("========================================")

    print("KeyFrames       :", config["keyframes"])
    print("Positive Prompt :", config["positive_prompt"])
    print("Negative Prompt :", config["negative_prompt"])
    print("Duration        :", config["duration"])
    print("Steps           :", config["steps"])
    print("Resolution      :", config["resolution"])
    print("CFG Scale       :", config["cfg_scale"])
    print("Seed            :", config["seed"])

    print()
    print("Hidden Defaults")
    print("----------------------------------------")

    print("latent_window_size       :", config["latent_window_size"])
    print("gs                       :", config["gs"])
    print("rs                       :", config["rs"])
    print("gpu_memory_preservation :", config["gpu_memory_preservation"])
    print("use_teacache            :", config["use_teacache"])
    print("mp4_crf                 :", config["mp4_crf"])
    print("teacache_threshold      :", config["teacache_threshold"])
    print("lora_file               :", config["lora_file"])
    print("lora_multiplier         :", config["lora_multiplier"])
    print("fp8_optimization        :", config["fp8_optimization"])

    print("========================================")

   



def test_adapter_config(
    timeline,
    gallery_images,
    positive_prompt,
    negative_prompt,
    duration,
    steps,
    resolution,
    cfg_scale,
    seed,
    runtime_path,
    webui_path
):
    config = build_render_config(
        timeline,
        positive_prompt,
        negative_prompt,
        duration,
        steps,
        resolution,
        cfg_scale,
        seed
    )

    motion_timeline = build_render_timeline(timeline,gallery_images)

    print()
    print("========================================")
    print("W6.4.2 : ADAPTER CONFIG TEST")
    print("========================================")

    print("KeyFrames :", config["keyframes"])

    print()
    print("Render Configuration")
    print("----------------------------------------")

    print("Positive Prompt :", config["positive_prompt"])
    print("Negative Prompt :", config["negative_prompt"])
    print("Duration        :", config["duration"])
    print("Steps           :", config["steps"])
    print("Resolution      :", config["resolution"])
    print("CFG Scale       :", config["cfg_scale"])
    print("Seed            :", config["seed"])

    print()
    print("Hidden Defaults")
    print("----------------------------------------")

    print("latent_window_size       :", config["latent_window_size"])
    print("gs                       :", config["gs"])
    print("rs                       :", config["rs"])
    print("gpu_memory_preservation :", config["gpu_memory_preservation"])
    print("use_teacache            :", config["use_teacache"])
    print("mp4_crf                 :", config["mp4_crf"])
    print("teacache_threshold      :", config["teacache_threshold"])
    print("lora_file               :", config["lora_file"])
    print("lora_multiplier         :", config["lora_multiplier"])
    print("fp8_optimization        :", config["fp8_optimization"])

    print()
    print("Creating MultiKeyAdapter...")
    adapter = MultiKeyAdapter()

    print("Adapter :", type(adapter).__name__)
    print("========================================")


    env = adapter.discover_runtime(runtime_path,webui_path)

    print()
    print("========================================")
    print("Runtime Discovery")
    print("========================================")
    print("Valid :", env.valid)
    print("Python :", env.python_executable)
    print("Demo :", env.demo_gradio)
    print("Working Directory :", env.working_directory)
    print("========================================")


    adapter.launch_runtime(env)

    adapter.wait_until_ready()

    adapter.connect_runtime()

    result = adapter.render_original(
    timeline=motion_timeline,
    prompt=config["positive_prompt"],
    negative_prompt=config["negative_prompt"],
    duration=config["duration"],
    steps=config["steps"],
    render_config=config
   )

    print()
    print("W6.4.3 RESULT")
    print("----------------------------------------")
    print(result)
    print("========================================")



def launch_webgui():
    with gr.Blocks(title="FramePack Multi-Keyframe", theme=gr.themes.Soft()) as demo:
        # ==========================================
        # Timeline Queue State
        # ==========================================

        timeline_images = gr.State([])
        with gr.Row():
            with gr.Column(scale=9):

                # ========================================
                # W2 : Timeline State
                # ========================================

                timeline_state = gr.State([])

                gr.Markdown("# 🎬 FramePack Multi-Keyframe")
                gr.Markdown("AI Animation Pipeline")
            with gr.Column(scale=2,min_width=220):
                 generate_button = gr.Button("▶ Generate Video",variant="primary",size="lg")
                
        with gr.Row():
            with gr.Column(scale=3):


                with gr.Group():
                    gr.Markdown("## 📁 Path Configuration")

                    runtime_path = gr.Textbox(label="FramePack Runtime Path",placeholder="Absolute path to FramePack runtime",interactive=True)

                    webui_path = gr.Textbox(label="FramePack WebUI Path",placeholder="Absolute path to FramePack WebUI",interactive=True)

                    output_path = gr.Textbox(label="Output Folder",placeholder="Absolute path for generated videos",interactive=True)

                    check_paths = gr.Button("🔍 Check Paths",variant="secondary")

                    path_status = gr.Textbox(label="Path Status",interactive=False,lines=5)
                
                with gr.Group():
                    gr.Markdown("## ⚙ Render Settings")

                    duration = gr.Number(label="Total Duration (seconds)",value=6,interactive=True)

                    steps = gr.Number(label="Steps",value=25,interactive=True)

                    resolution = gr.Dropdown(["360p", "540p", "720p"],value="720p",label="Resolution",interactive=True)

                    cfg_scale = gr.Number(label="CFG Scale",value=1.0,interactive=True)

                    seed = gr.Number(label="Seed",value=-1,interactive=True)
                
                with gr.Group():
                    gr.Markdown("## 📊 Status")
                    gr.Textbox(label="Status",value="Idle",interactive=False)
                    gr.Slider(0,100,value=0,label="Progress",interactive=False)
            with gr.Column(scale=8):
                with gr.Group():
                    with gr.Row():

                         gr.Markdown("## 🎞 Keyframe Timeline")

                         clear_queue = gr.Button("🗑 Clear Queue",variant="stop",size="sm",min_width=140)

                    gr.Markdown("Timeline displays images selected from the Image Library.")

                    timeline_html = gr.HTML("""
                    <div style="
                    height:140px;
                    border:2px dashed #666;
                    border-radius:12px;
                    padding:15px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:#999;
                    font-size:20px;
                    ">

                    No keyframes selected.

                    Click an image below to begin.

                    </div>
                    """)

                    
                with gr.Group():
                    gr.Markdown("## ▶ Animation Preview")
                    preview = gr.Video(label="",height=500,show_label=False,interactive=False,autoplay=False,elem_id="animation_preview")

                    gr.Markdown(
                    """
                    <center>

                    Current Preview

                    <small>
                    After rendering, the generated video will appear here.
                    </small>

                    </center>
                    """
                    )


                with gr.Group():
                    with gr.Row():
                        gr.Markdown("## 🖼 Image Library")
                        refresh=gr.Button("Refresh")
                   
                    gr.Markdown("Click an image to add it to the Keyframe Timeline.")

                    search_library = gr.Textbox(placeholder="🔍 Search images...",show_label=False)
                    library = gr.Gallery(label="",columns=6,rows=4,height=430,object_fit="contain",allow_preview=False)
                    library.select(fn=add_to_timeline,inputs=[timeline_images, library],outputs=[timeline_images, timeline_html, library])
                    clear_queue.click(fn=clear_timeline,outputs=[timeline_images, timeline_html])
                    check_paths.click(fn=validate_paths,inputs=[runtime_path,webui_path,output_path],outputs=path_status)

                    def load_library():
                        return scan_images()
                    
                    refresh.click(load_library,outputs=library)

                    demo.load(load_library,outputs=library)
                    
            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("## 💬 Prompts")
                    positive_prompt = gr.Textbox(label="Positive Prompt",lines=8)
                    negative_prompt = gr.Textbox(label="Negative Prompt",lines=5)

                generate_button.click(fn=test_adapter_config,inputs=[timeline_images,library,positive_prompt,negative_prompt,duration,steps,resolution,cfg_scale,seed,runtime_path, webui_path],outputs=[])


                with gr.Group():
                    gr.Markdown("## 📋 Timeline Summary")

                    


                    summary = gr.Markdown(
                    """
                    ### Project Statistics

                    | Property | Value |
                    |----------|-------|
                    | Images | 0 |
                    | Segments | 0 |
                    | Total Duration | 0 sec |
                    | Duration / Segment | -- |
                    | Resolution | 720p |
                    | Steps | 25 |
                    | Status | Ready |

                    ---

                    ### Workflow

                    ① Select images from the Image Library.

                    ↓

                    ② Arrange them inside the Timeline.

                    ↓

                    ③ Configure prompts and render settings.

                    ↓

                    ④ Click **Generate Video**.
                    """
                    )

    demo.launch()



def add_to_timeline(evt: gr.SelectData, timeline, gallery_images):

    image_index = evt.index

    print()
    print("========================================")
    print("TIMELINE CLICK")
    print("========================================")
    print("Index :", image_index)

    if image_index not in timeline:
        timeline.append(image_index)

    print("Timeline :", timeline)

    print("Timeline :", timeline)

    render_timeline = reverse_render_sequence(timeline)

    print("Render Timeline:", render_timeline)

    motion_timeline = build_render_timeline(timeline,gallery_images)

    print()
    print("========================================")
    print("W6.2 : MotionTimeline")
    print("========================================")

    print("KeyFrames :", len(motion_timeline.keyposes))

    for pose in motion_timeline.keyposes:
      print(pose.index + 1,"-",pose.image_path)

      print("========================================")


    adapter = MultiKeyAdapter()

    print()
    print("========================================")
    print("W6.3 : MULTIKEY ADAPTER TEST")
    print("========================================")

    print("Adapter :", type(adapter).__name__)

    print("KeyFrames received:")

    for pose in motion_timeline.keyposes:
      print(pose.index + 1,"-",pose.image_path)
      print("========================================")


    return (timeline,refresh_timeline(timeline, gallery_images),gr.update(selected=None))




def clear_timeline():
    return [], refresh_timeline([], [])





def refresh_timeline(timeline, gallery_images):

    if not timeline:
        return """
        <div style="
        height:140px;
        border:2px dashed #666;
        border-radius:12px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#999;
        font-size:20px;
        ">
        No keyframes selected.<br>
        Click an image below.
        </div>
        """

    html = """
    <div style="
    display:flex;
    gap:12px;
    align-items:center;
    overflow-x:auto;
    padding:8px;
    ">
    """

    for i, image_index in enumerate(timeline):
        img = gallery_images[image_index]
        image_path = img[0]
        print("=" * 60)
        print(img)
        print(type(img))
        print(img[0])
        print("=" * 60)
        html += f"""
        <div style="
        min-width:100px;
        padding:10px;
        border:1px solid #666;
        border-radius:8px;
        text-align:center;
        background:#2c2c2c;
        ">

          <img
          src="/gradio_api/file={quote(str(Path(image_path).resolve()).replace(os.sep, '/'))}"
          style="
          width:90px;
          height:90px;
          object-fit:contain;
          border-radius:8px;
          "/>

          <div style="margin-top:6px">
            Pose {image_index + 1}
          </div>

        </div>
        """

        if i < len(timeline) - 1:
            html += """
            <div style="font-size:26px">
            ➜
            </div>
            """

    html += "</div>"

    return html

if __name__=="__main__":

    

    launch_webgui()
