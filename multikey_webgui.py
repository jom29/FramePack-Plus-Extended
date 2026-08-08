import os
from pathlib import Path
import gradio as gr
from urllib.parse import quote

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
                gr.Button("▶ Generate Video",variant="primary",size="lg")
        with gr.Row():
            with gr.Column(scale=3):


                with gr.Group():
                    gr.Markdown("## 📁 Path Configuration")

                    runtime_path = gr.Textbox(label="FramePack Runtime Path",placeholder="Absolute path to FramePack runtime")

                    webui_path = gr.Textbox(label="FramePack WebUI Path",placeholder="Absolute path to FramePack WebUI")

                    output_path = gr.Textbox(label="Output Folder",placeholder="Absolute path for generated videos")
                
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

                    def load_library():
                        return scan_images()
                    
                    refresh.click(load_library,outputs=library)

                    demo.load(load_library,outputs=library)
                    
            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("## 💬 Prompts")
                    gr.Textbox(label="Positive Prompt",lines=8)
                    gr.Textbox(label="Negative Prompt",lines=5)
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
