import torch
import torchvision.models as models

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
print(model)
# Set the model to evaluation mode
model.eval()

# Create example data to trace the model
example = torch.rand(1, 3, 224, 224)

# Trace the model
traced_script_module = torch.jit.trace(model, example)

# Save the traced model
# traced_script_module.save("resnet50_model.pt")