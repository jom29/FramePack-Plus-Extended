import os
import sys
import subprocess
from runtime_environment import RuntimeEnvironment

import time
import requests
from gradio_client import Client, handle_file


import os

class MultiKeyAdapter:

    def __init__(self):

        self.process = None

    # --------------------------------------------------




    def connect_runtime(self):

     print()

     print("========================================")
     print("Connecting To Runtime")
     print("========================================")

     self.client = Client("http://127.0.0.1:17861")

     print()

     print("Connected Successfully")

     print()

     print("========================================")
     print("API Information")
     print("========================================")

     self.client.view_api()



    # --------------------------------------------------

    def discover_runtime(

      self,

      runtime_root,

      webui_root

     ):

        env = RuntimeEnvironment()

        env.runtime_root = runtime_root

        #
        # Candidate Python locations
        #

        python_candidates = [

            os.path.join(

                runtime_root,

                "python",

                "python.exe"

            ),

            os.path.join(

                runtime_root,

                "system",

                "python",

                "python.exe"

            ),

            os.path.join(

                runtime_root,

                "venv",

                "Scripts",

                "python.exe"

            ),

            os.path.join(

                runtime_root,

                ".venv",

                "Scripts",

                "python.exe"

            ),

        ]

        for path in python_candidates:

            if os.path.exists(path):

                env.python_executable = path

                break

        
    # demo_gradio
    # (Use the WebUI folder selected by the user)
    #

        demo_path = os.path.join(

         webui_root,

         "demo_gradio.py"

        )

        if os.path.exists(demo_path):

         env.demo_gradio = demo_path

         env.working_directory = webui_root

        else:

         print()

         print("ERROR")
         print("demo_gradio.py not found!")
         print(demo_path)

        env.valid = (

        env.python_executable != ""

        and

      env.demo_gradio != ""

     )

        return env







    def render_motion(
      self,
      timeline,
      prompt,
      negative_prompt,
      duration,
      resolution,
      steps,
      runtime_root,
      webui_root,
    output_folder
    ):

      print()
      print("========================================")
      print("MultiKey Adapter")
      print("========================================")

      print()
      print("Runtime")
      print(runtime_root)

      print()
      print("WebUI")
      print(webui_root)

      print()
      print("Output")
      print(output_folder)

      print()
      print("----------------------------------------")
      print("MotionTimeline Received")
      print("----------------------------------------")

      print()
      print("KeyFrames :", len(timeline.keyposes))

      for pose in timeline.keyposes:
        print(
            pose.index + 1,
            "-",
            pose.image_path.name
        )

      print()

      env = self.discover_runtime(
        runtime_root,
        webui_root
     )

      print()
      print("========================================")
      print("Runtime Discovery")
      print("========================================")

      print()
      print("Valid")
      print(env.valid)

      print()
      print("Python")
      print(env.python_executable)

      print()
      print("Demo")
      print(env.demo_gradio)

      print()
      print("Working Directory")
      print(env.working_directory)

      self.launch_runtime(env)

      self.wait_until_ready()

      self.connect_runtime()

    # --------------------------------------------------
    # Research Pipeline
    # --------------------------------------------------

      self.render_direct(
        timeline,
        prompt,
        negative_prompt,
        duration,
        steps
     )

      return

    # --------------------------------------------------
    # Original FramePack Pipeline
    # (Disabled while doing research)
    # --------------------------------------------------

    # self.render_original(
    #     timeline,
    #     prompt,
    #     negative_prompt,
    #     duration,
    #     steps
    # )



    #------------------------------------------------
    def render_original(
      self,
      timeline,
      prompt,
      negative_prompt,
      duration,
      steps
  ):




      print()
      print("========================================")
      print("Original Render Request")
      print("========================================")




      print()

      print()

      print("========================================")
      print("Direct Research Pipeline")
      print("========================================")

      print()

     
     

    def render_direct(
   
     self,
     timeline,
     prompt,
     negative_prompt,
     duration,
     steps
     ):
      

      from PIL import Image
      import numpy as np
      import torch

      from diffusers_helper.hunyuan import vae_encode



      from PIL import Image
      import numpy as np
      import torch

      print()
      print("========================================")
      print("Direct Research Pipeline")
      print("========================================")

      # --------------------------------------------------
      # Load Images
      # --------------------------------------------------

      loaded_images = []

      print()
      print("========================================")
      print("Loading Images")
      print("========================================")

      for pose in timeline.keyposes:

        image = Image.open(
            pose.image_path
        ).convert("RGB")

        loaded_images.append(image)

        print(
            f"[{pose.index}]",
            image.size
        )

      # --------------------------------------------------
      # Convert To NumPy
      # --------------------------------------------------

      numpy_images = []

      print()
      print("========================================")
      print("Converting To NumPy")
      print("========================================")

      for i, image in enumerate(loaded_images):

        array = np.array(image)

        numpy_images.append(array)

        print(
            f"[{i}]",
            array.shape,
            array.dtype
        )

      # --------------------------------------------------
      # Convert To Torch Tensor
      # --------------------------------------------------

      tensor_images = []

      print()
      print("========================================")
      print("Converting To Torch Tensor")
      print("========================================")

      for image in numpy_images:

        image_pt = torch.from_numpy(image).float() / 127.5 - 1

        image_pt = image_pt.permute(
            2,
            0,
            1
        )[None, :, None]

        tensor_images.append(image_pt)

      for i, tensor in enumerate(tensor_images):

        print(
            f"[{i}]",
            tuple(tensor.shape),
            tensor.dtype
        )

      print()
      print("========================================")
      print("Research Stage Complete")
      print("========================================")
      print(f"Loaded Images : {len(loaded_images)}")
      print(f"NumPy Arrays  : {len(numpy_images)}")
      print(f"Torch Tensors : {len(tensor_images)}")

      return


















    #----------------------------------------------------------------
    def wait_until_ready(self):

      print(">>> Entered wait_until_ready()")

      print()

      print("========================================")
      print("Waiting For Runtime")
      print("========================================")

      while True:

        #
        # Runtime crashed?
        #

        if self.process.poll() is not None:

            raise RuntimeError(

                "Official Runtime exited unexpectedly."

            )

        #
        # Try HTTP
        #

        try:

            response = requests.get(

                "http://127.0.0.1:17861",

                timeout=2

            )

            if response.status_code == 200:

                print()

                print("Runtime Ready")

                print()

                return

        except Exception:

            pass

        print("Waiting...")

        time.sleep(1)




    def launch_runtime(self, env):

     print()

     print("========================================")
     print("Launching Runtime")
     print("========================================")

     self.process = subprocess.Popen(

     [

        env.python_executable,

        env.demo_gradio,

        "--server",

        "127.0.0.1",

        "--port",

        "17861"

     ],

     cwd=env.working_directory

     )

     print()

     print("PID :", self.process.pid)

     import time

     time.sleep(3)

     print()

     print("========================================")
     print("Runtime Status")
     print("========================================")

     if self.process.poll() is None:

       print("Runtime is still running.")
       print("Milestone 4 PASSED.")
       return
     
     else:

       print("Runtime exited immediately.")
       print("Exit Code :", self.process.returncode)

       print()

       print("============= Runtime Log =============")

     