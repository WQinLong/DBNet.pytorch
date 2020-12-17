import torch
from models import build_model

model_path = './output/DBNet_opensource_fintune_resnet18_FPN_DBHead/checkpoint/model_best.pth'
device = torch.device("cpu")
checkpoint = torch.load(model_path, map_location=device)

config = checkpoint['config']
config['arch']['backbone']['pretrained'] = False
model = build_model(config['arch'])

model.load_state_dict(checkpoint['state_dict'], strict=False)
f = model_path.replace('.pth', '.state_dict.pth')
torch.save(checkpoint['state_dict'], f)

model.float().eval()
# todo fuse()
# todo Hardswish
# todo 删除thread map

img = torch.rand(1, 3, 1024, 1024)
img = img.to(device)

model.eval()
model.to(device)
y = model(img, None)
import time

num = 1
begin = time.time()
for i in range(num):
    y = model(img, None)
end = time.time()
print((end - begin) / 100)

try:
    print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    f = model_path.replace('.pth', '.norm.torchscript.pt')  # filename
    ts = torch.jit.trace(model, img)
    ts.save(f)
    print('TorchScript export success, saved as %s' % f)
except Exception as e:
    print('TorchScript export failure: %s' % e)

try:
    import onnx

    # import numpy as np
    # img = np.zeros(1, 3, 1024, 1024)
    img = img.to(torch.uint8)
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = model_path.replace('.pth', '.norm.onnx')  # filename
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch_size', 2: 'width', 3: 'height'},
                    'output': {0: 'batch_size', 2: 'width', 3: 'height'}}  # adding names for better debugging
    torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=input_names,
                      output_names=output_names, dynamic_axes=dynamic_axes)

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)
