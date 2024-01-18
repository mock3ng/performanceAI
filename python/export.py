from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision
from time import sleep
from torch.utils.mobile_optimizer import optimize_for_mobile

print("")
print("**********************")
print("")
print("----------------------- Model Converting! -----------------------")
print("Please wait, system is running")
print("")
print("**********************")
print("")

imgsize = (<img-size>)  # Image size must be first height after width!!!

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

class WrapperModel(torch.nn.Module):

    def __init__(self, model: torch.nn.Module):
    	super().__init__()
    	self.model = model

    def forward(self, input_tensor: torch.Tensor, conf_thres: float = 0.0):
        prediction = self.model(input_tensor)
        bs = prediction.shape[0]  
        nc = (prediction.shape[1] - 4)  
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  
        xc = prediction[:, 4:mi].amax(1) > conf_thres
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction): 
            print(f"prediction: {xi}")
            x = x.transpose(0, -1)[xc[xi]] 
            print(x.shape)
            if not x.shape[0]:
                continue
            box, cls, mask = x.split((4, nc, nm), 1)
            box = xywh2xyxy(box)
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            n = x.shape[0]  
            if not n: 
                continue
            x = x[x[:, 4].argsort(descending=True)[:30000]]
            c = x[:, 5:6] * 0  
            boxes, scores = x[:, :4] + c, x[:, 4]
            return x, boxes, scores

model = YOLO("<YOLOMODEL-path>", task="detect")
model.export(format="torchscript", imgsz=imgsize)
print("")
print("**********************")
print("")
print("----------------------- Model Converted! -----------------------")
print(f"Format: torchscript, Image Size= {imgsize}")
print("Please wait now model converting PyTorch Lite format for Android...")
print("")
print("**********************")
print("")
print("Please wait....")
print("")
sleep(3)
torchmodel = torch.load(<save-torchscript>)  
wrapper_model= WrapperModel(torchmodel)
scriptedmodel2 = torch.jit.script(wrapper_model)
scriptedmodel2.save(<save-torchscript>)
"""
mobilemodel = optimize_for_mobile(scriptedmodel2) # This option is optional, if you want you can remove this line!
mobilemodel._save_for_lite_interpreter(<save-torchscriptlite>)
"""
print("")
print("**********************")
print("")
print("----------------------- Model Converted! -----------------------")
print(f"Format: PyTorch Lite, Image Size= {imgsize}")
print("Model saved!")
print("")
print("**********************")
print("")
