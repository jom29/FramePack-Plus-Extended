
"""
multikey_webgui.py
Milestone W1 - WebGUI Skeleton
"""

import gradio as gr

APP_TITLE="🎬 FramePack Multi-Keyframe"
APP_SUBTITLE="AI Animation Pipeline"

def launch_webgui():
   with gr.Blocks(title="FramePack Multi-Keyframe",theme=gr.themes.Soft(),fill_height=True,) as demo:

              gr.Markdown("# 🎬 FramePack Multi-Keyframe")

              gr.Markdown("AI Video Generation using Multi-Keyframe Timeline")

              # ======================================================
              # Main Layout
              # ======================================================

              with gr.Row():

               # ==========================================
               # LEFT PANEL
               # ==========================================

               with gr.Column(scale=3):

                 # ==========================================
                 # PROJECT CONFIGURATION
                 # ==========================================

                  gr.Markdown("## 📁 Project Configuration")
                  runtime_path = gr.Textbox(label="FramePack Runtime",placeholder="FramePack Runtime Folder")

                  output_path = gr.Textbox(label="Output Folder",placeholder="Output Folder")

                  gr.Markdown("---")

                  # ==========================================
                  # RENDER SETTINGS
                  # ==========================================

                  gr.Markdown("## ⚙️ Render Settings")

                  duration = gr.Number(label="Total Duration (Seconds)",value=6.0)

                  steps = gr.Number(label="Steps",value=25)

                  resolution = gr.Dropdown(label="Resolution",choices=["360p","540p","720p"],value="720p")

                  cfg = gr.Number(label="CFG Scale",value=1.0)

                  seed = gr.Number(label="Seed",value=-1)

                  gr.Markdown("---")

                  generate_btn = gr.Button("🎬 Generate Video",variant="primary",size="lg")

           

               # ==========================================
               # CENTER PANEL
               # ==========================================

               with gr.Column(scale=7):

                    gr.Markdown("## Keyframe Timeline")

                    gr.Markdown("---")

                    gr.Markdown("## Animation Preview")

                    gr.Markdown("---")

                    gr.Markdown("## Image Library")

                # ==========================================
                # RIGHT PANEL
                # ==========================================

               with gr.Column(scale=3):

                    gr.Markdown("## Prompt")

                    gr.Markdown("---")

                    gr.Markdown("## Render Status")

      

   demo.launch()

if __name__=="__main__":
    launch_webgui()
