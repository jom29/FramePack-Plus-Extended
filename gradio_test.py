from gradio_client import Client, handle_file

print("Connecting...")
client = Client("http://127.0.0.1:7860")
print("Connected")

print("Calling /process...")

result = client.predict(

    input_image=handle_file(r"C:\Users\ADMIN\Pictures\KeyFrames\projects\demo\images\Frame1.png"),
    end_image=handle_file(r"C:\Users\ADMIN\Pictures\KeyFrames\projects\demo\images\Frame2.png"),

    prompt="test",
    n_prompt="",

    seed=31337,
    total_second_length=1,
    latent_window_size=9,
    steps=5,
    cfg=1.0,
    gs=10.0,
    rs=0.0,
    gpu_memory_preservation=6,
    use_teacache=True,
    mp4_crf=16,
    resolution=512,
    teacache_threshold=0.15,

    lora_file=None,
    lora_multiplier=0.8,
    fp8_optimization=False,

    api_name="/process"

)

print("Render Finished")
print(result)