from __future__ import absolute_import

def print_params(model):
  """
  Print the number of trainable and untrainable parameters in the model.
  
  Args:
    model: The model to print the parameters of.
  """
  trainable = 0 # number of trainable parameters
  untrainable = 0 # number of untrainable parameters
  trainable_size = 0 # size in mb of trainable parameters

  print("Layers:")
  for name, param in model.named_parameters():
    print(f"- {name} of size {param.size()} -> {'trainable' if param.requires_grad else 'untrainable'}")

    if param.requires_grad: # if the parameter requires a gradient
      trainable += param.numel() # increment the number of trainable parameters
      trainable_size += param.numel() * param.element_size() # increment the size of trainable parameters
    else: # otherwise
      untrainable += param.numel() # increment the number of untrainable parameters

  print(f"\nTrainable parameters: {trainable}")
  print(f"Untrainable parameters: {untrainable}")
  print(f"Total parameters: {trainable + untrainable}")
  print(f"Percent trainable: {100 * trainable / (trainable + untrainable)}%")
  percentage = trainable_size / 1e6
  print(f"Size of trainable parameters: {percentage:.2f} mb")
