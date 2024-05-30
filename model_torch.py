import utils
import fire
import torch


class HandWritingModule(torch.nn.Module):

  def __init__(self):
    super().__init__()

    # input layer: 28x28 pixels = 784, outputs 16 features
    self.input_layer = torch.nn.Linear(28 * 28, 16, bias=True)

    # output layer: 16 input features, outputs 10 classes (0-9)
    self.output_layer = torch.nn.Linear(16, 10, bias=True)

    # ReLU activation function - max(0, x)
    self.activation = torch.nn.ReLU()

  def load(self, model_file):
    self.load_state_dict(torch.load(model_file))

  def save(self, model_file):
    torch.save(self.state_dict(), model_file)

  def forward(self, x):
    x = self.input_layer(x)
    x = self.activation(x)
    x = self.output_layer(x)
    return x
  
def main():
  # 28x28 grayscale images (normalised 0-1) of handwritten digits (0-9)
  # Labels are one-hot encoded ([1, 2, 3] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  train_images, train_labels, test_images, test_labels = utils.load_datasets()

  model = HandWritingModule()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  loss_function = torch.nn.MSELoss()

  num_epochs = 500
  batch_size = 32

  for epoch in range(num_epochs):
    permutation = torch.randperm(train_images.size()[0])

    for i in range(0, train_images.size(0), batch_size):
      optimizer.zero_grad()

      indices = permutation[i:i + batch_size]
      batch_x, batch_y = train_images[indices], train_labels[indices]

      outputs = model(batch_x)
      loss = loss_function(outputs, batch_y)
      loss.backward()
      optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Test the model
    with torch.no_grad():
      test_outputs = model(test_images)
      test_loss = loss_function(test_outputs, test_labels)
      print(f"Validation Loss: {test_loss.item()}")

      # Calculate accuracy
      predictions = torch.argmax(test_outputs, dim=1)
      correct = (predictions == torch.argmax(test_labels, dim=1)).sum().item()
      accuracy = correct / test_labels.size(0)
      print(f"Accuracy: {accuracy * 100:.2f}%")

    print()

  # Save the model
  model.save(f"handwriting_model_torch_{num_epochs}.pt")

if __name__ == "__main__":
  fire.Fire(main)
