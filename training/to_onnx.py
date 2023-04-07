import torch.onnx
from unet import UNet
model = UNet(3,1)


checkpoint = torch.load("unet/best_140.pth", map_location= 'cuda')["model_state_dict"]
model.load_state_dict(checkpoint, strict=False)
model = model.eval()
x = torch.randn(1, 3, 80, 160, requires_grad=True)
# y = model(x)
# print(y.size())
torch.onnx.export(model, x, "unet_pytorch_8x16.onnx" ,opset_version=11, verbose=True, input_names = ['input'], output_names = ['output'])
