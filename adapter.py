
import os
import sys
import time
import threading
import subprocess
import requests

from gradio_client import Client, handle_file


class FramePackAdapter:

    def __init__(self, runtime_root, webui_root):
        self.runtime_root = runtime_root
        self.webui_root = webui_root
        self.process = None
        self.client = None

    def launch(self):
        embedded = os.path.join(self.runtime_root, "system", "python", "python.exe")
        venv = os.path.join(self.runtime_root, "venv", "Scripts", "python.exe")

        if os.path.exists(embedded):
            python = embedded
        elif os.path.exists(venv):
            python = venv
        else:
            python = sys.executable

        demo = os.path.join(self.webui_root, "demo_gradio.py")

        self.process = subprocess.Popen(
            [python, demo, "--server", "127.0.0.1", "--port", "7860", "--inbrowser"],
            cwd=self.webui_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def wait_until_ready(self):
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("FramePack exited unexpectedly.")
            try:
                if requests.get("http://127.0.0.1:7860", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)

    def connect(self):
        self.client = Client("http://127.0.0.1:7860")

    def _server_log(self):
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            print("[SERVER]", line.rstrip())

    def render_phase(self, start_image, end_image, prompt,
                     negative_prompt, duration, resolution, steps):

        threading.Thread(target=self._server_log, daemon=True).start()

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
            resolution=resolution,
            teacache_threshold=0.15,
            lora_file=None,
            lora_multiplier=0.8,
            fp8_optimization=False,
            api_name="/process"
        )

        last = None
        while True:
            try:
                s = job.status()
                if str(s) != last:
                    print(s)
                    last = str(s)
            except Exception:
                break

            if job.done():
                break

            time.sleep(2)

        return job.result(timeout=None)

    def shutdown(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
