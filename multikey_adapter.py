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

      self.render_original(
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

      start_image = str(timeline.keyposes[0].image_path)

      end_image = str(timeline.keyposes[-1].image_path)

      print()

      print("Start")

      print(start_image)

      print()

      print("End")

      print(end_image)

      print()

      print("Submitting Request...")







      print()
      print("========================================")
      print("Original Render Request")
      print("========================================")

      start_image = str(timeline.keyposes[0].image_path)

      end_image = str(timeline.keyposes[-1].image_path)

      print("Start :", start_image)
      print("End   :", end_image)

      print()
      print("Submitting request...")

      import time

      print()
      print("[1] Building Payload")

      payload = {

      "input_image": {

        "path": start_image

      },

      "end_image": {

        "path": end_image

      },

       "prompt": prompt,

      "n_prompt": negative_prompt,

      "seed": 31337,

      "total_second_length": duration,

      "latent_window_size": 9,

      "steps": steps,

      "cfg": 1.0,

      "gs": 10.0,

      "rs": 0.0,

      "gpu_memory_preservation": 6,

      "use_teacache": True,

      "mp4_crf": 16

     }

      











      job = self.client.submit(

       input_image=handle_file(start_image),

       end_image=handle_file(end_image),

       prompt=prompt,

      n_prompt=negative_prompt,

      seed=31337,

      total_second_length=duration,

      latent_window_size=9,

      steps=steps,

      cfg=1.0,

      gs=10.0,

      rs=0.0,

      gpu_memory_preservation=6,

      use_teacache=True,

      mp4_crf=16,

      resolution=640,

      teacache_threshold=0.15,

      lora_file=None,

      lora_multiplier=0.8,

      fp8_optimization=False,

      api_name="/process"

     )

      print()
      print("Job Submitted")

      last = None

      while True:

       status = job.status()

       if str(status) != last:

         print(status)

         last = str(status)

       if job.done():

         break

      time.sleep(2)

      print()
      print("Retrieving Result...")

      result = job.result(timeout=None)

      print()
      print("========================================")
      print("Render Finished")
      print("========================================")

      print(result)





















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

     