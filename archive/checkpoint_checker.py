import torch

# Load the checkpoint on CPU
checkpoint = torch.load("/u/pmaldonadocatala/hyena-dna/weights/weights.ckpt", map_location=torch.device('cpu'))

# Print the keys in the checkpoint
print(checkpoint.keys())
print(checkpoint['state_dict'].keys())  # Check the contents of the state_dict