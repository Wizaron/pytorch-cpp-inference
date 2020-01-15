import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True)

# Evaluation mode
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

def export_cpu(model, example):
    model = model.to("cpu")
    example = example.to("cpu")

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save traced model
    traced_script_module.save("resnet_model_cpu.pth")

def export_gpu(model, example):
    model = model.to("cuda")
    example = example.to("cuda")

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Save traced model
    traced_script_module.save("resnet_model_gpu.pth")

export_cpu(model, example)

if torch.cuda.is_available():
    export_gpu(model, example)
