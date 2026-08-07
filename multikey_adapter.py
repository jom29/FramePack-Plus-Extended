import platform
import shutil
import os
import sys
import subprocess
from runtime_environment import RuntimeEnvironment

import threading
import time
import requests
from gradio_client import Client, handle_file


import os

class MultiKeyAdapter:

    def __init__(self):

      self.process = None

      self.heartbeat_thread = None

      self.heartbeat_running = False

    # --------------------------------------------------



    def heartbeat_loop(self):

      print("[Heartbeat] Thread Started")

      while self.heartbeat_running:

        try:

            response = requests.get(
                "http://127.0.0.1:17861",
                timeout=3
            )

            print(
                "[Heartbeat]",
                response.status_code
            )

        except Exception as ex:

            print(
                "[Heartbeat] Runtime unreachable:",
                ex
            )

        time.sleep(10)

      print("[Heartbeat] Thread Stopped")




    def start_heartbeat(self):

        if self.heartbeat_running:
         return

        self.heartbeat_running = True

        self.heartbeat_thread = threading.Thread(
          target=self.heartbeat_loop,
          daemon=True
        )

        self.heartbeat_thread.start()



    def stop_heartbeat(self):

      self.heartbeat_running = False

      if self.heartbeat_thread is not None:

        self.heartbeat_thread.join(timeout=2)





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

       

        python_candidates = []

        if platform.system() == "Windows":

          python_candidates = [

          os.path.join(runtime_root, "python", "python.exe"),

          os.path.join(runtime_root, "system", "python", "python.exe"),

          os.path.join(runtime_root, "venv", "Scripts", "python.exe"),

          os.path.join(runtime_root, ".venv", "Scripts", "python.exe"),
        ]

        else:

          current_python = sys.executable

          if current_python:
            python_candidates.append(current_python)

          python3 = shutil.which("python3")
          if python3:
            python_candidates.append(python3)

          python = shutil.which("python")
          if python:
            python_candidates.append(python)



        # Search for an existing Python executable
        for path in python_candidates:

          if os.path.exists(path):

           env.python_executable = path

           break

          # Fallback to the currently running Python
        if env.python_executable == "":

           env.python_executable = sys.executable









        
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

      self.start_heartbeat()

      try:

        self.render_original(
          timeline,
          prompt,
          negative_prompt,
          duration,
          steps
        )

      finally:

        self.stop_heartbeat()

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





      



      # ==========================================================
      # POC-2 : Runtime Frame Collection
      # ==========================================================

      frame_collection = []

      for pose in timeline.keyposes:

        frame_collection.append(
        str(pose.image_path)
      )


      start_image = frame_collection[0]

      end_image = frame_collection[-1]

      print()

      print("========================================")
      print("POC-2 : Adapter Frame Collection")
      print("========================================")

      print()

      print("Total Frames :", len(frame_collection))

      print()

      for index, frame in enumerate(frame_collection):

       print(f"Frame {index}")
       print(frame)
       print()








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

      frame_collection=[
      {
        "path": frame
      }
      for frame in frame_collection
      ],

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
      api_name="/process_multikey"
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
     print("Executable :", repr(env.python_executable))
     print("Script     :", repr(env.demo_gradio))
     print("WorkingDir :", repr(env.working_directory))
     print()

     print("========================================")
     print("Launching Runtime")
     print("========================================")

     runtime_root = os.path.dirname(os.path.dirname(os.path.dirname(env.python_executable)))
     system_dir = os.path.join(runtime_root, "system")

     proc_env = os.environ.copy()

     hf_cache = os.path.join(runtime_root,"webui","hf_download")

     proc_env["HF_HOME"] = hf_cache

     proc_env["HF_HUB_CACHE"] = os.path.join(hf_cache,"hub")

     proc_env["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache,"hub")

     proc_env["PATH"] = (
         os.path.join(system_dir, "git", "bin")
         + ";"
         + os.path.join(system_dir, "python")
         + ";"
         + os.path.join(system_dir, "python", "Scripts")
         + ";"
         + proc_env.get("PATH", "")
     )

     proc_env["SKIP_VENV"] = "1"
     proc_env["PY_LIBS"] = os.path.join(system_dir, "python", "Lib") + ";" + os.path.join(system_dir, "python", "Lib", "site-packages")
     proc_env["PY_PIP"] = os.path.join(system_dir, "python", "Scripts")
     proc_env["PIP_INSTALLER_LOCATION"] = os.path.join(system_dir, "python", "get-pip.py")

     self.process = subprocess.Popen(
         [
             env.python_executable,
             env.demo_gradio,
             "--server",
             "127.0.0.1",
             "--port",
             "17861",
         ],
         cwd=env.working_directory,
         env=proc_env
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
