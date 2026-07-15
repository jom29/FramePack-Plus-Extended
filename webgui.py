import gradio as gr

from controller import Controller


class FramePackWebGUI:

    def __init__(self):

        self.controller = Controller()

        self.build_interface()

    # --------------------------------------------------
    # Build Interface
    # --------------------------------------------------

    def build_interface(self):

        with gr.Blocks(title="FramePack Plus Extended") as self.app:

            self.build_header()
            self.build_workspace()

    def build_workspace(self):

                with gr.Row():

                   with gr.Column(scale=1):

                      self.build_phase_navigation()

                   with gr.Column(scale=5):

                       self.build_phase_editor()

                       gr.Markdown("---")

                self.bind_events()





    def build_phase_editor(self):

        self.build_phase_header()

        gr.Markdown("---")

        self.build_preview_panel()

        gr.Markdown("---")

        self.build_image_library()

        gr.Markdown("---")

        self.build_controls()


    def build_phase_header(self):

      self.phase_title = gr.Markdown(f"""## Currently Editing### {self.controller.get_current_phase().name}""")


    def build_phase_navigation(self):

        gr.Markdown("## 🎬 Phases")

        self.phase_selector = gr.Radio(

           choices=self.controller.get_phase_names(),

           value=self.controller.get_current_phase().name,

           label=None

        )

        gr.Markdown("---")

        self.add_phase_button = gr.Button("➕ Add Phase",variant="secondary")

        self.delete_phase_button = gr.Button("🗑 Delete Phase",variant="secondary")





    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    def build_header(self):

        gr.Markdown(
            """
# 🎬 FramePack Plus Extended

AI Animation Pipeline
            """
        )

    # --------------------------------------------------
    # Preview Panel
    # --------------------------------------------------

    def build_preview_panel(self):

        with gr.Row():

            with gr.Column(scale=1):

                gr.Markdown("## Start Keyframe")

                self.start_preview = gr.Image(
                    label="Preview",
                    interactive=False,
                    height=320
                )

                self.start_filename = gr.Textbox(
                    label="Selected Image",
                    value="None",
                    interactive=False
                )

            with gr.Column(scale=1):

                gr.Markdown("## End Keyframe")

                self.end_preview = gr.Image(
                    label="Preview",
                    interactive=False,
                    height=320
                )

                self.end_filename = gr.Textbox(
                    label="Selected Image",
                    value="None",
                    interactive=False
                )

    # --------------------------------------------------
    # Image Library
    # --------------------------------------------------

    def build_image_library(self):

        gr.Markdown("## 📁 Image Library")

        gr.Markdown("""Images are automatically loaded from`projects/demo/images`""")

        with gr.Row():

            self.refresh_button = gr.Button(
                "🔄 Refresh Library",
                variant="secondary",
                scale=0
            )

            self.search_box = gr.Textbox(placeholder="Search image...",show_label=False,scale=4)

        self.gallery = gr.Gallery(value=self.controller.get_images(),columns=6,rows=2,height=300,object_fit="contain",allow_preview=True,show_label=False)

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------

    def build_controls(self):

        with gr.Row():

            self.set_start_button = gr.Button(
                "⬅ Set Selected as Start",
                variant="primary"
            )

            self.set_end_button = gr.Button(
                "Set Selected as End ➡",
                variant="primary"
            )

    # --------------------------------------------------
    # Events
    # --------------------------------------------------

    def bind_events(self):

      self.phase_selector.change(

        fn=self.on_phase_changed,

        inputs=self.phase_selector,

        outputs=self.phase_title

    )

      self.add_phase_button.click(

        fn=self.on_add_phase,

        outputs=[

            self.phase_selector,

            self.phase_title

        ]

    )

      self.delete_phase_button.click(

        fn=self.on_delete_phase,

        outputs=[

            self.phase_selector,

            self.phase_title

        ]

    )




    def on_phase_changed(self, phase_name):

        names = self.controller.get_phase_names()

        index = names.index(phase_name)

        self.controller.select_phase(index)

        return f"""
    ## Currently Editing

    ### {self.controller.get_current_phase().name}
    """





    def on_add_phase(self):

        self.controller.add_phase()

        names = self.controller.get_phase_names()

        current = self.controller.get_current_phase().name

        return (

        gr.update(

            choices=names,

            value=current

        ),

        f"""
## Currently Editing

### {current}
"""

    )





    def on_delete_phase(self):

       self.controller.delete_current_phase()

       names = self.controller.get_phase_names()

       current = self.controller.get_current_phase().name

       return (

        gr.update(

            choices=names,

            value=current

        ),

        f"""
## Currently Editing

### {current}
"""

    )






    # --------------------------------------------------
    # Launch
    # --------------------------------------------------

    def launch(self):

        self.app.launch()


if __name__ == "__main__":

    gui = FramePackWebGUI()

    gui.launch()