# Toy Handwritten Digit Recognition

This is a toy project to recognize handwritten digits using a simple neural network. The neural network is implemented from scratch using only Python. The dataset used is the MNIST dataset.

## Usage

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

Run the following command to train the neural network:

```bash
python model_from_scratch.py
```

Then to run inference on a test image, run the following command:

```bash
python3 inference --input-image <path-to-image>
```

Or you can use the Gradio interface to run inference:

```bash
python3 app.py
```
