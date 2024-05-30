import torch
import gradio as gr
import numpy as np
from inference import inference_from_scratch, load_from_scratch, inference_from_torch, load_torch

scratch_model = load_from_scratch()
torch_model = load_torch()

def inference(image_array):
  scratch = inference_from_scratch(scratch_model, image_array)
  torch_model_output = inference_from_torch(torch_model, torch.tensor([image_array], dtype=torch.float32))
  
  return f"From-Scratch Model: {scratch}, Torch Model: {torch_model_output}"

def recognize_digit(image):
  # Resize image to 28x28
  image = image['composite'].resize((28, 28))

  # Save to disk
  image.save("digit.jpg")

  # Convert image to numpy array
  image_array = np.array(image)

  # Normalize the pixel values
  image_array = image_array / 255.0

  # Make small values black
  image_array[image_array < 0.3] = 0.0

  # 1D array
  image_array = image_array.flatten()

  # Run inference
  digit = inference(image_array.tolist())
  return str(digit)

with gr.Blocks() as demo:
  gr.Markdown("## Handwritten Digit Recognition")
  canvas = gr.Sketchpad(
    brush=gr.Brush(colors=["white"]),
    canvas_size=(650, 650),
    image_mode="L",
    type="pil",
    layers=False,
  )
  output = gr.Textbox(label="Recognized Digit")
  button = gr.Button("Submit")
  button.click(recognize_digit, inputs=canvas, outputs=output)

demo.launch()
