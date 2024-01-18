In the dynamic landscape of machine learning, the transformation of YOLO or PyTorch models into TorchScript has gained prominence, showcasing a methodology that advocates for leveraging these models in C++ rather than Python. This paradigm shift not only brings about improvements in efficiency but also promises a significant boost in execution speed, thereby revolutionizing the way we deploy and utilize deep learning models. As the demand for real-time applications and responsiveness continues to surge, the transition to C++ usage underscores a strategic move toward achieving optimal performance and resource utilization in the realm of computer vision and artificial intelligence. This essay aims to delve into the rationale behind this shift, exploring the intricacies of TorchScript conversion and elucidating the advantages it confers upon applications, ultimately advocating for the adoption of C++ as a preferred environment for deploying YOLO or PyTorch models.

In this article, we will convert our model into a torchscript model with Python and get our outputs in cpp. These models can be YOLO Models and pytorch models. If you do not have pytorch models, you can convert your models to .pt extension using the pytorch API.

Loading the YOLO Model and presenting the results in Python :

Result of using the same model in CPP with torchscript conversion :

We were able to speed it up more than 10 times. Please note that this time includes loading the model into memory.
PYTHON CONVERT MODEL :

    Importing Necessary Libraries:

from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision
from time import sleep
from torch.utils.mobile_optimizer import optimize_for_mobile

In this section, the required libraries and modules are imported. This includes the Ultralytics YOLO library, PyTorch, and related modules.

    Initialization for Model Conversion:

imgsize = (1920, 1088)  # Image size must be first height after width!!!

Setting the image size as a tuple (height, width) for later use in model conversion.

    Defining xywh2xyxy Function:

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

This function is designed to modify the coordinate format obtained from the model’s output.

    Defining the WrapperModel Class:

class WrapperModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_tensor: torch.Tensor, conf_thres: float = 0.0):
        # Forward pass function, organizes the output, and filters above a certain confidence threshold.
        # Then takes the top 30000 objects with the best scores.
        # Returns scores, boxes, and classes as output.
        # This function serves the purpose of organizing the output of models.

A class that organizes the output of the model. This class selects objects with the best scores above a certain confidence threshold.

    Model Conversion and Saving:

model = YOLO("<YOLOMODEL-path>", task="detect")
model.export(format="torchscript", imgsz=imgsize)

Loading the Ultralytics YOLO model and converting it to the specified format (in this case, torchscript).

    Saving the TorchScript Model:

torchmodel = torch.load("/home/cha0/Desktop/cpp/ktscpp/best.torchscript")  
wrapper_model = WrapperModel(torchmodel)
scriptedmodel2 = torch.jit.script(wrapper_model)
scriptedmodel2.save("/home/cha0/Desktop/cpp/ktscpp/bestcpp.torchscript")

Saving the model for PyTorch Lite conversion.

    Printing the Results:

print("----------------------- Model Converted! -----------------------")
print(f"Format: torchscript, Image Size= {imgsize}")
print("Please wait now model converting PyTorch Lite format for Android...")
print("")
print("**********************")
print("")
print("Please wait....")
print("")
sleep(3)

Informative messages indicating the successful conversion of the model.

    Converting to PyTorch Lite Format and Saving (Optional):

mobilemodel = optimize_for_mobile(scriptedmodel2)  # This option is optional, you can remove it if not needed!
# mobilemodel._save_for_lite_interpreter("./models/penisM_Y_FHD_0210.ptl")

Optionally, creating a mobile-optimized model and saving it in PyTorch Lite format.

    Printing the Results (Continuation):

print("----------------------- Model Converted! -----------------------")
print(f"Format: PyTorch Lite, Image Size= {imgsize}")
print("Model saved!")
print("")
print("**********************")
print("")

Informative messages indicating the successful conversion of the model to PyTorch Lite format.

In summary, this Python script utilizes the Ultralytics YOLO library to convert a YOLO model to the torchscript format and then saves the output model. The script appears to be intended for making the model usable on Android devices.
CPP USING:
Header and Libraries

#include <iostream>
#include <list>
#include <string>
#include <torch/script.h>
#include <torch/types.h>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

In this section, necessary C++ libraries, as well as the header files for PyTorch and OpenCV, are included.
load_input_image Function

torch::Tensor load_input_image(const std::string& filename, int width, int height) {
  cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

  if (image.empty()) {
    std::cerr << "Error: " << filename << std::endl;
    return torch::Tensor();
  }  cv::resize(image, image, cv::Size(width, height));  torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kByte);  tensor = tensor.permute({2, 0, 1}).to(torch::kFloat32).div(255.0);  tensor = tensor.unsqueeze(0);  return tensor;
}

This function loads an image from a specified file, resizes it to the specified width and height, converts it to a tensor, and scales pixel values between 0 and 1. The resulting tensor is prepared as input for the model.
parse_row Function

std::vector<float> parse_row(torch::Tensor row, int img_width, int img_height) {
  float xc = row[0].item<float>();
  float yc = row[1].item<float>();
  float w = row[2].item<float>();
  float h = row[3].item<float>();

  float x1 = (xc - w / 2) / 1920 * img_width;
  float y1 = (yc - h / 2) / 1088 * img_height;
  float x2 = (xc + w / 2) / 1920 * img_width;
  float y2 = (yc + h / 2) / 1088 * img_height;  int n = row.size(0);  torch::Tensor probs = row.narrow(0, 4, n - 4);  auto maxProb = probs.max().item<float>();  return {x1, y1, x2, y2, maxProb};
}

This function takes a row from the model’s output tensor and processes it by converting normalized coordinates and adding the probability of the highest probable class. It returns a vector: [x1, y1, x2, y2, maxProb].
Box Operations

float intersection(torch::Tensor box1, torch::Tensor box2);
float unioner(torch::Tensor box1, torch::Tensor box2);
float iou(torch::Tensor box1, torch::Tensor box2);

These three functions are used to compute the intersection area, union area, and IoU (Intersection over Union) ratio between two bounding boxes.
main Function

int main() {
  //...
}

This function is the main entry point of the program. It coordinates the execution flow of the program.
Necessary Variables and Loading Input Image

std::vector<std::string> yolo_classes = {<classes name "","","">};

cv::Mat img;
img = cv::imread(<image-path>);
cv::Size size = img.size();
int img_width = size.width;
int img_height = size.height;
torch::Tensor input = load_input_image(<image-path>, <image-size>);

In this part, a vector named yolo_classes is defined to hold the names of the classes. Then, an image is loaded, and its dimensions are obtained. The image is converted to the required format using the load_input_image function.
Loading the Model and Running it

torch::jit::script::Module module = torch::jit::load(<torchscript-mode-path>);
auto start_time = std::chrono::high_resolution_clock::now();
torch::Tensor output = module.forward({input}).toTensor();
auto end_time = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
output = output[0].transpose(0,1);
torch::Tensor row = output[0];

The model is loaded using the torch::jit::script::Module class. The prepared input image is passed to the model, and the output is obtained. The execution time of the model is measured and printed. The output tensor is transposed for further processing, and a row from the output is extracted.
Processing and Filtering the Output

std::vector<std::vector<float>> filteredRows;
for(int i=0; i<output.size(0); i++) {
  torch::Tensor row = output[i];
  std::vector<float> parsedRow = parse_row(row, img_width, img_height);
  if(parsedRow[4] > 0.7) {
    filteredRows.push_back(parsedRow);
  }
}

A loop is used to iterate over the output tensor. Each row is processed using the parse_row function, and if the probability of the detected object is greater than a certain threshold (0.7 in this case), the row is added to the filteredRows vector.
Visualization and Printing Results

for(auto& row : filteredRows) {
  for(int i=0; i<row.size(); i+=5) {
    float x1 = row[i];
    float y1 = row[i+1]; 
    float x2 = row[i+2];
    float y2 = row[i+3];
    //auto  label = std::to_string(row[i+4]);

    cv::rectangle(img, cv::Point(x1,y1), cv::Point(x2,y2),  
                  cv::Scalar(0,255,0), 2);    // cv::putText(img, label, cv::Point(x1,y1), 
    //           cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0));
  }
}

Finally, the coordinates of the filtered rows are used to draw rectangles on the original image. The rectangles represent the detected objects. The image is then displayed.

Make sure to replace placeholders like <classes name "","","">, <image-path>, <image-size>, and <torchscript-mode-path> with the actual values for your use case.

Here, on the cpp outputs, if you cannot find a proper label coordinate on your photo, float intersection(torch::Tensor box1, torch::Tensor box2);
float unioner(torch::Tensor box1, torch::Tensor box2);
float iou(torch::Tensor box1, torch::Tensor box2); You can change its functions. You can find main.cpp, CMakeLists.txt, convert.py, requirements.txt on my github page along with all the codes. Don’t forget to follow me, I will soon have articles explaining the use of Python with Onnx models, Openvino models.
