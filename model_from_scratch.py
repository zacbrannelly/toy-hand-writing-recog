import random
import math
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from typing import List

# Some notes on the backpropagation algorithm:

# linear_activation = weight * second_last_activation + bias
# last_activation = non_linear_func(linear_activation)
# cost = (last_activation - label) ** 2

# The gradient to be applied to the weight is: delta(cost) / delta(weight)
# So we use the chain rule and the relations above to calculate it:

# delta(cost) / delta(weight)
# = delta(cost) / delta(last_activation) *
#   delta(last_activation) / delta(linear_activation) * 
#   delta(linear_activation) / delta(weight)

# First we calculate the derivatives of each of the components:

# delta(cost) / delta(last_activation) = 2 * (last_activation - label)
# delta(last_activation) / delta(linear_activation) = delta_non_linear_func(linear_activation)
# delta(linear_activation) / delta(weight) = second_last_activation

# Here we give the above derivatives some names:

# output_gradient = 2 * (last_activation - label)
# activation_gradient = delta_non_linear_func(linear_activation)

# Then we calculate the gradient for the weight and apply it using SGD:

# gradient = output_gradient * activation_gradient * second_last_activation
# SGD: weight = weight - learning_rate * gradient

# Next we calculate the input of the next layer's backward pass (the second last layer):

# output_gradient_for_second_last_layer = delta(cost) / delta(second_last_activation)

# output_gradient_for_second_last_layer = 
#   delta(cost) / delta(last_activation) * 
#   delta(last_activation) / delta(linear_activation) *
#   delta(linear_activation) / delta(second_last_activation)

# We have already derived the first two components (output_gradient & activation_gradient), we just need the last one:
# linear_activation = weight * second_last_activation + bias
# Therefore:
# delta(linear_activation) / delta(second_last_activation) = weight

# Now we put it all together:
# output_gradient_for_second_last_layer = output_gradient * activation_gradient * weight

# Then we loop and apply the same process to the N last layers, we just replace output_gradient with the gradient_for_N_last_layer.

# To reiterate succinctly, for the output layer (last layer) ONLY, the input (output_gradient) to the backward pass function is:
# = 2 * (last_activation - label)

# For the remaining hidden layers before the output layer, the input (output_gradient) for their backward pass function is:
# = output_gradient_from_next_layer * activation_gradient_from_next_layer * weight_from_next_layer

# Hence for layer N the input is calculated in the N+1 layer backward pass and passed to the N layer backward pass as input.

# ----------------------------

# delta(cost) / delta(bias)
# = delta(cost) / delta(last_activation) *
#   delta(last_activation) / delta(linear_activation) * 
#   delta(linear_activation) / delta(bias)

# delta(linear_activation) / delta(bias) = 1

# gradient = output_gradient * activation_gradient
# SGD: bias = bias - learning_rate * gradient

LEARNING_RATE = 0.01

class Model:

  def __init__(self, layers):
    self.layers = layers

  def forward(self, input_batch: List[List[float]]):
    activations = input_batch
    for layer in self.layers:
      activations = layer.forward(activations)

    return activations

  def loss(self, predictions_per_batch: List[List[float]], labels_per_batch: List[List[float]]):
    loss_per_batch: List[List[float]] = []
    loss_gradient_per_batch: List[List[float]] = []

    for i in range(len(predictions_per_batch)):
      predictions = predictions_per_batch[i]
      labels = labels_per_batch[i]

      loss = []
      loss_gradient = []
      for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[i]

        # Mean Squared Error
        loss.append((prediction - label) ** 2)
        
        # Loss gradient
        loss_gradient.append(2 * (prediction - label))

      loss_per_batch.append(loss)
      loss_gradient_per_batch.append(loss_gradient)

    return loss_per_batch, loss_gradient_per_batch

  def backward(self, gradients_per_batch: List[List[float]]):
    for layer in reversed(self.layers):
      gradients_per_batch = layer.backward(gradients_per_batch)


class Layer:
  def __init__(self, in_features=10, out_features=10):
    self.in_features = in_features
    self.out_features = out_features
    self.neurons = [Neuron(in_features) for _ in range(out_features)]

    # Store the previous activations for back propagation
    self.previous_activations = []
  
  def forward(self, previous_activations: List[List[float]]):
    # Store the previous activations for back propagation
    self.previous_activations = previous_activations

    batch_size = len(previous_activations)
    activations: List[List[float]] = [[0.0] * len(self.neurons) for _ in range(batch_size)]

    for neuron_idx, neuron in enumerate(self.neurons):
      predictions_per_batch = neuron.forward(previous_activations)
      for batch_idx in range(batch_size):
        activations[batch_idx][neuron_idx] = predictions_per_batch[batch_idx]

    return activations
  
  def backward(self, output_gradients_per_batch: List[List[float]]):
    batch_size = len(output_gradients_per_batch)
    input_gradients = [[0.0] * self.in_features for _ in range(batch_size)]

    for neuron_idx in range(len(self.neurons)):
      neuron = self.neurons[neuron_idx]

      # Get the gradient that maps to the current neuron for each example in the batch
      neuron_output_gradient_per_batch: List[float] = []
      for batch_idx in range(batch_size):
        output_gradient = output_gradients_per_batch[batch_idx][neuron_idx]
        neuron_output_gradient_per_batch.append(output_gradient)
    
      gradients_per_batch = neuron.backward(neuron_output_gradient_per_batch, self.previous_activations)

      for batch_idx in range(batch_size):
        for k in range(self.in_features):
          input_gradients[batch_idx][k] += gradients_per_batch[batch_idx][k]

    return input_gradients


class Neuron:

  def __init__(self, in_features=10):
    # Initialize weights similar to PyTorch's default initialization:
    k = math.sqrt(1.0 / in_features)
    self.weight = [random.uniform(-k, k) for _ in range(in_features)]
    self.bias = random.uniform(-k, k)

    # Store the linear activations (for each example in the input batch) for back propagation
    self.linear_activation = []

    self.activation_fct = lambda x: 1 / (1 + math.exp(-x))
    self.delta_activation_fct = lambda x: self.activation_fct(x) * (1 - self.activation_fct(x))

    self.gradient = []

  def forward(self, previous_activations_per_batch: List[List[float]]) -> List[float]:
    final_activations = []
    self.linear_activation = []

    # Run forward pass over each example in the batch
    for batch_idx in range(len(previous_activations_per_batch)):
      previous_activations = previous_activations_per_batch[batch_idx]

      linear_activation = 0.0
      for weight_idx in range(len(previous_activations)):
        linear_activation += self.weight[weight_idx] * previous_activations[weight_idx]

      linear_activation += self.bias

      # Store linear activation for back propagation
      self.linear_activation.append(linear_activation)

      # Apply activation function
      activation = self.activation_fct(linear_activation)

      final_activations.append(activation)
    
    return final_activations
  
  def backward(self, output_gradient_per_batch: List[float], previous_activations_per_batch: List[List[float]]) -> List[List[float]]:
    self.gradient = []
    
    input_gradients_per_batch = []
    accumulated_gradients = [0.0] * len(self.weight)
    accumulated_bias_gradient = 0.0

    # Run backward pass over each example in the batch
    # Just track the gradients for the weights and biases, no updates yet.
    batch_size = len(output_gradient_per_batch)
    for batch_idx in range(batch_size):
      output_gradient = output_gradient_per_batch[batch_idx]
      previous_activations = previous_activations_per_batch[batch_idx]
      linear_activation = self.linear_activation[batch_idx]
      
      activation_gradient = self.delta_activation_fct(linear_activation)
      error = output_gradient * activation_gradient

      input_gradients = []
      for weight_idx in range(len(self.weight)):
        weight = self.weight[weight_idx]
        previous_activation = previous_activations[weight_idx]

        # delta(cost_one_example) / delta(weight)
        gradient = error * previous_activation

        # Accumulate the gradients for the batch
        accumulated_gradients[weight_idx] += gradient

        # Generate input gradients for the backward pass of the previous layer.
        input_gradient = error * weight
        input_gradients.append(input_gradient)

      # delta(cost_one_example) / delta(bias)
      accumulated_bias_gradient += error

      input_gradients_per_batch.append(input_gradients)

    # Update the weights and biases using the average gradients
    for weight_idx in range(len(self.weight)):
      self.weight[weight_idx] -= LEARNING_RATE * accumulated_gradients[weight_idx]
      self.gradient.append(accumulated_gradients[weight_idx])
    self.bias -= LEARNING_RATE * accumulated_bias_gradient

    return input_gradients_per_batch


def main():
  # Load the MNIST dataset
  train_input, train_expected, validation_input, validation_expected = utils.load_datasets_as_lists()

  # Create the model
  model = Model([
    Layer(in_features=28 * 28, out_features=16),
    Layer(in_features=16, out_features=10),
  ])

  # Generate batches
  batch_size = 32
  input_batches = [train_input[i:i + batch_size] for i in range(0, len(train_input), batch_size)]
  expected_batches = [train_expected[i:i + batch_size] for i in range(0, len(train_expected), batch_size)]
  batches = list(zip(input_batches, expected_batches))

  num_epochs = 100
  average_loss = 0.0

  train_loss = []
  val_loss = []

  # Epoch = one iteration over all of the training data.
  for epoch in range(num_epochs):
    random.shuffle(batches)

    step_index = 0
    for input_batch, expected_batch in tqdm(batches, desc=f"Epoch {epoch + 1}/{num_epochs}"):
      # Forward propagation
      predictions = model.forward(input_batch)
      losses, output_gradients = model.loss(predictions, expected_batch)

      # Backward propagation
      model.backward(output_gradients)
    
      if step_index % 100 == 0:
        # Calculate the average loss over the epoch
        average_loss = sum([sum(loss) / len(loss) for loss in losses]) / len(losses)

        # Validation
        validation_predictions = model.forward(validation_input)
        validation_losses, _ = model.loss(validation_predictions, validation_expected)
        validation_loss = sum([sum(loss) / len(loss) for loss in validation_losses]) / len(validation_losses)

        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {average_loss}")
        print(f"Validation Loss: {validation_loss}")

        train_loss.append(average_loss)
        val_loss.append(validation_loss)

        # Validate the model (MNIST)
        accuracy = 0.0
        for i in range(len(validation_predictions)):
          prediction = validation_predictions[i]
          expected = validation_expected[i]

          if prediction.index(max(prediction)) == expected.index(max(expected)):
            accuracy += 1
        
        accuracy /= len(validation_predictions)
        print(f"Validation Accuracy: {accuracy * 100}%")
      
      step_index += 1

    # Show the graph at the end of the epoch
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
  main()
