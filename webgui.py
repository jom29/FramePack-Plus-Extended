import gradio as gr

from pathlib import Path

from controller import Controller
from render_engine import RenderEngine
from settings import load_paths, save_paths
from video_stitcher import VideoStitcher


class FramePackWebGUI:

    def __init__(self):

        self.controller = Controller()

        self.controller.initialize_project()

        self.render_engine = RenderEngine(

          self.controller

        )

        self.stitcher = VideoStitcher()


        self.build_interface()


    def on_save_settings(

      self,

      positive,

      negative,

      duration,

      steps,

      resolution

     ):

      self.controller.commit_inspector(

        positive,

        negative,

        duration,

        steps,

        resolution

    )


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

        self.build_video_preview()

        self.build_preview_panel()

        gr.Markdown("---")

        self.build_image_library()

        gr.Markdown("---")

        self.build_controls()


    def build_phase_header(self):

       self.phase_title = gr.Markdown(
         f"""
       ## Currently Editing

       ### {self.controller.get_current_phase().name}
"""
    )


    def build_phase_navigation(self):

        gr.Markdown("## 🎬 Phases")

        self.save_settings_button = gr.Button(

        "💾 Save Settings",

        variant="primary"

       )
     

        self.phase_selector = gr.Radio(

           choices=self.controller.get_phase_names(),

           value=self.controller.get_current_phase().name,

           label=None

        )

        gr.Markdown("---")

        self.add_phase_button = gr.Button("➕ Add Phase",variant="secondary")

        self.delete_phase_button = gr.Button("🗑 Delete Phase",variant="secondary")


        gr.Markdown("---")

        gr.Markdown("## ⚙ Render Settings")

        
        self.settings_mode = gr.Radio(

          choices=[

           "Global",

           "Local"

        ],

           value="Global",

           label="Settings Mode"

        )





        self.positive_prompt = gr.Textbox(

            label="Positive Prompt",

            lines=4,

            value=self.controller.get_positive_prompt()

        )

        self.negative_prompt = gr.Textbox(

            label="Negative Prompt",

            lines=4,

            value=self.controller.get_negative_prompt()

        )

        with gr.Row():

          with gr.Row():

           self.duration = gr.Number(

           label="Duration (Seconds)",

           value=self.controller.get_duration(),

           precision=1
        )




        with gr.Row():

            self.steps = gr.Number(

                label="Steps",

                value=self.controller.get_steps(),

                precision=0

            )

            self.resolution = gr.Dropdown(

                label="Resolution",

                choices=[

                    480,

                    720,

                    1080

                ],

                value=self.controller.get_resolution()

            )

        gr.Markdown("---")

        self.generate_button = gr.Button(

            "🚀 Generate Batch",

            variant="primary"

        )

        self.generate_current_button = gr.Button(

         "🎬 Generate Current Phase",

          variant="secondary"

        )


        self.progress = gr.Slider(

            minimum=0,

            maximum=100,

            value=0,

            label="Progress",

            interactive=False

        )

        self.status = gr.Textbox(

            label="Status",

            value="Idle",

            interactive=False

        )



        gr.Markdown("---")

        gr.Markdown("## 🎞 Output")

        self.stitch_button = gr.Button(

        "🎞 Merge Clips",

         variant="secondary",

        interactive=True

       )
        


        self.generate_button.click(

         fn=self.on_generate,

         inputs=[

      self.positive_prompt,

      self.negative_prompt,

      self.duration,

      self.steps,

      self.resolution

    ],

    outputs=[

        self.progress,

        self.status

    ]

 )



        self.generate_current_button.click(

         fn=self.on_generate_current_phase,

        inputs=[

        self.positive_prompt,

        self.negative_prompt,

        self.duration,

        self.steps,

        self.resolution

    ],

    outputs=[

        self.progress,

        self.status

    ]

  )






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
    # Video Preview
    # --------------------------------------------------

    def build_video_preview(self):

     gr.Markdown("## 🎬 Animation Preview")

     self.video_preview = gr.Video(

        value=self.controller.get_current_phase_video(),

        interactive=False,

        autoplay=False,

        show_label=False,

        height=420

     )

     self.video_status = gr.Markdown(

     self.controller.get_current_phase_video_status()

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

     gr.Markdown(
        "Images are automatically loaded from `projects/demo/images`"
     )

     # Toolbar
     with gr.Row():

        self.refresh_button = gr.Button(
            "🔄 Refresh Library",
            variant="secondary",
            scale=0
        )

        self.search_box = gr.Textbox(
            placeholder="Search image...",
            show_label=False,
            scale=4
        )

     # Gallery occupies the full width below the toolbar
     self.gallery = gr.Gallery(
        value=self.controller.get_images(),
        columns=6,
        rows=None,
        height=500,
        object_fit="contain",
        allow_preview=True,
        show_label=False,
        elem_id="image_gallery"
    )
            
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

        outputs=[

        self.phase_title,

        self.video_preview,

        self.video_status,

        self.start_preview,

        self.start_filename,

        self.end_preview,

        self.end_filename,

        self.positive_prompt,

        self.negative_prompt,

        self.duration,

        self.steps,

        self.resolution

       ]

      )

      self.add_phase_button.click(

        fn=self.on_add_phase,

     outputs=[

      self.phase_selector,

      self.phase_title,

      self.video_preview,

      self.video_status,

      self.start_preview,

      self.start_filename,

      self.end_preview,

      self.end_filename,

      self.positive_prompt,

      self.negative_prompt,

      self.duration,

      self.steps,

      self.resolution

        ]


       )
      



      self.delete_phase_button.click(

        fn=self.on_delete_phase,

        outputs=[

        self.phase_selector,

        self.phase_title,

        self.video_preview,

        self.video_status,

        self.start_preview,

        self.start_filename,

        self.end_preview,

        self.end_filename,

        self.positive_prompt,

        self.negative_prompt,

        self.duration,

        self.steps,

        self.resolution
      ]
     )
      

      self.set_start_button.click(

         fn=self.on_set_start,

         outputs=[

         self.start_preview,

         self.start_filename

        ]

     )


      
      self.gallery.select(

         fn=self.on_gallery_selected,

         outputs=[]

     )
      
      
      self.refresh_button.click(

      fn=self.on_refresh_library,

      outputs=[

        self.gallery

    ]

   )
      

      self.set_end_button.click(

         fn=self.on_set_end,

         outputs=[

         self.end_preview,

         self.end_filename

                ]

      )




      self.stitch_button.click(

      fn=self.on_merge

      )


      self.settings_mode.change(

       fn=self.on_settings_mode_changed,

      inputs=self.settings_mode,

      outputs=[

        self.positive_prompt,

        self.negative_prompt,

        self.duration,

        self.steps,

        self.resolution
        ]
       )
      
      self.save_settings_button.click(

      fn=self.on_save_settings,

      inputs=[

        self.positive_prompt,

        self.negative_prompt,

        self.duration,

        self.steps,

        self.resolution

      ],

      outputs=[]

      )









    def on_set_start(self):

        self.controller.set_start_image()

        image = self.controller.get_start_image()

        if image:

           filename = Path(image).name

        else:

          filename = "None"

        return (

        image,

        filename

    )



    def on_set_end(self):

       self.controller.set_end_image()

       image = self.controller.get_end_image()

       if image:

        filename = Path(image).name

       else:

        filename = "None"

       return (

        image,

        filename

    )


    def refresh_phase_editor(self):

     phase = self.controller.get_current_phase()

     print()
     print("========================================")
     print("Refreshing Phase Editor")
     print("========================================")
     print("Phase :", phase.name)

     video = self.controller.get_current_phase_video()

     video = gr.update(
       value=video
     )

     video_status = self.controller.get_current_phase_video_status()

     start_image = self.controller.get_start_image()
     end_image = self.controller.get_end_image()

     start_name = (
        Path(phase.start_image).name
        if phase.start_image
        else "None"
    )

     end_name = (
        Path(phase.end_image).name
        if phase.end_image
        else "None"
    )

     title = f"""
       ## Currently Editing

      ### {phase.name}
      """

     values = self.controller.load_inspector()

     return (

       title,

       video,

       video_status,

       start_image,

       start_name,

       end_image,

       end_name,

       values[0],

       values[1],

       values[2],

       values[3],

       values[4]

   )


    def on_gallery_selected(self, evt: gr.SelectData):

      image_path = evt.value["image"]["path"]

      self.controller.select_image(image_path)

      print(image_path)


    def on_refresh_library(self):

     print()
     print("========================================")
     print("Refreshing Image Library")
     print("========================================")

     images = self.controller.get_images()

     print(f"Found {len(images)} images.")

     return gr.update(

        value=images

    )
    
    
      
       



    def on_phase_changed(self, phase_name):

      names = self.controller.get_phase_names()

      index = names.index(phase_name)

      self.controller.select_phase(index)

      return self.refresh_phase_editor()





    def on_add_phase(self):

        self.controller.add_phase()

        names = self.controller.get_phase_names()

        current = self.controller.get_current_phase().name

        editor = self.refresh_phase_editor()

        return (

         gr.update(

        choices=names,

        value=current

     ),

      *editor

    )







    def on_delete_phase(self):

     self.controller.delete_current_phase()

     names = self.controller.get_phase_names()

     current = self.controller.get_current_phase().name

     editor = self.refresh_phase_editor()

     return (

        gr.update(
            choices=names,
            value=current
        ),

        *editor

    )




   # --------------------------------------------------
   # Generate Batch
   # --------------------------------------------------

    def on_generate(

     self,

     positive_prompt,

     negative_prompt,

     duration,

     steps,

    resolution

    ):

      self.controller.set_positive_prompt(

        positive_prompt

     )

      self.controller.set_negative_prompt(

        negative_prompt

    )

      self.controller.set_duration(

        duration

    )

      self.controller.set_steps(

        steps

    )

      self.controller.set_resolution(

        resolution

    )


      self.controller.print_render_queue()


      self.render_engine.render_project(

        progress_callback=self.update_progress

    )

      return (

        100,

        "Finished"

    )



    def on_generate_current_phase(

     self,

     positive_prompt,

     negative_prompt,

     duration,

     steps,

     resolution

 ):

     self.controller.set_positive_prompt(

        positive_prompt

    )

     self.controller.set_negative_prompt(

        negative_prompt

    )

     self.controller.set_duration(

        duration

    )

     self.controller.set_steps(

        steps

    )

     self.controller.set_resolution(

        resolution

    )

     self.render_engine.render_current_phase(

        progress_callback=self.update_progress

    )

     return (

        100,

        "Finished"

    )






    def on_settings_mode_changed(

      self,

      mode

     ):

      self.controller.set_settings_mode(

        mode.lower()

     )

      values = self.controller.load_inspector()

      return (

            values[0],

            values[1],

            values[2],

            values[3],

            values[4]

           )

      print("--------------------------------")

      print(

        "Settings Mode :",

        self.controller.get_settings_mode()

     )

      print("--------------------------------")







    def on_merge(self):

     self.stitcher.merge()



    def update_progress(

      self,

      percent,

      message

    ):

      print(

        f"{percent}% - {message}"

    )



    # --------------------------------------------------
    # Launch
    # --------------------------------------------------

    def launch(self):

        self.app.launch(

        server_name="0.0.0.0",

        server_port=7860

    )


if __name__ == "__main__":

    runtime_root, webui_root = load_paths()

    print()
    print("========================================")
    print(" FramePack Plus Extended Setup")
    print("========================================")
    print()

    runtime = input(
        f"FramePack Runtime\n[{runtime_root}]\n> "
    ).strip()

    if runtime:
        runtime_root = runtime

    webui = input(
        f"\nFramePack Plus WebUI\n[{webui_root}]\n> "
    ).strip()

    if webui:
        webui_root = webui

    save_paths(runtime_root, webui_root)

    gui = FramePackWebGUI()

    gui.launch()