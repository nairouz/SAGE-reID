from __future__ import absolute_import

from collections import OrderedDict


from collections import defaultdict
import math

def compute_cluster_entropy_and_weights(cluster_labels, cluster_cams):
    """
    cluster_labels: list or array of cluster indices (length N)
    cluster_cams: list of camera IDs (length N)
    Returns:
        weight_dict: cluster_id -> w_l (float)
    """
    cluster_to_cam_counts = defaultdict(lambda: defaultdict(int))

    for label, cam in zip(cluster_labels, cluster_cams):
        if label == -1:
            continue
        cluster_to_cam_counts[label][cam] += 1

    weight_dict = {}
    for cluster, cam_counts in cluster_to_cam_counts.items():
        total = sum(cam_counts.values())
        probs = [count / total for count in cam_counts.values()]
        entropy = -sum(p * math.log(p + 1e-8) for p in probs)
        weight = math.log(entropy + 1.0)
        weight_dict[cluster] = weight

    return weight_dict


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



def state_dict_cleaner(state_dict, strip_prefix="module."):
    """
    Return a copy of `state_dict` where all keys starting with `strip_prefix`
    have that prefix removed.

    Example:
    'module.layer1.weight' -> 'layer1.weight'
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(strip_prefix):
            new_key = k[len(strip_prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict
