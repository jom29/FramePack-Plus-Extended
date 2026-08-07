import os
import gradio as gr

PROJECTS_ROOT="projects"

def image_folder(project):
    return os.path.join(PROJECTS_ROOT, project, "images")

def scan_images(project):
    folder=image_folder(project)
    if not os.path.isdir(folder):
        return []
    exts=(".png",".jpg",".jpeg",".webp")
    return [os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]

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
                    gr.Textbox(label="FramePack Runtime Path")
                    gr.Textbox(label="Projects Root",value="projects")
                    gr.Textbox(label="Current Project",value="demo")
                    gr.Textbox(label="Output Folder")
                with gr.Group():
                    gr.Markdown("## ⚙ Render Settings")
                    gr.Number(label="Total Duration (seconds)",value=6)
                    gr.Number(label="Steps",value=25)
                    gr.Dropdown(["360p","540p","720p"],value="720p",label="Resolution")
                    gr.Number(label="CFG Scale",value=1.0)
                    gr.Number(label="Seed",value=-1)
                with gr.Group():
                    gr.Markdown("## 📊 Status")
                    gr.Textbox(label="Status",value="Idle",interactive=False)
                    gr.Slider(0,100,value=0,label="Progress",interactive=False)
            with gr.Column(scale=8):
                with gr.Group():
                    gr.Markdown("## 🎞 Keyframe Timeline")
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
                    project=gr.Textbox(value="demo",visible=False)
                    gr.Markdown("Click an image to add it to the Keyframe Timeline.")

                    search_library = gr.Textbox(placeholder="🔍 Search images...",show_label=False)
                    library = gr.Gallery(label="",columns=6,rows=4,height=430,object_fit="contain",allow_preview=True,preview=True)
                    library.select(fn=add_to_timeline,inputs=[timeline_images, library],outputs=[timeline_images, timeline_html])
                    def load_library(p):
                        return scan_images(p)
                    refresh.click(load_library,project,library)
                    demo.load(load_library,project,library)
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

    return timeline, refresh_timeline(timeline, gallery_images)





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
          src="/gradio_api/file={image_path.replace('\\','/')}"
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
