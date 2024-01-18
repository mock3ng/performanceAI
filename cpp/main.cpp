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
torch::Tensor load_input_image(const std::string& filename, int width, int height) {
  cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

  if (image.empty()) {
    std::cerr << "Eror: " << filename << std::endl;
    return torch::Tensor();
  }

  cv::resize(image, image, cv::Size(width, height));

  torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kByte);

  tensor = tensor.permute({2, 0, 1}).to(torch::kFloat32).div(255.0);

  tensor = tensor.unsqueeze(0);

  return tensor;
}

std::vector<float> parse_row(torch::Tensor row, int img_width, int img_height) {
  float xc = row[0].item<float>();
  float yc = row[1].item<float>();
  float w = row[2].item<float>();
  float h = row[3].item<float>();

  float x1 = (xc - w / 2) / 1920 * img_width;
  float y1 = (yc - h / 2) / 1088 * img_height;
  float x2 = (xc + w / 2) / 1920 * img_width;
  float y2 = (yc + h / 2) / 1088 * img_height;

  int n = row.size(0);

  torch::Tensor probs = row.narrow(0, 4, n - 4);

  auto maxProb = probs.max().item<float>();

  return {x1, y1, x2, y2, maxProb};
}

float intersection(torch::Tensor box1, torch::Tensor box2) {
  float box1_x1 = box1[0].item<float>();
  float box1_y1 = box1[1].item<float>();
  float box1_x2 = box1[2].item<float>();
  float box1_y2 = box1[3].item<float>();

  float box2_x1 = box2[0].item<float>();
  float box2_y1 = box2[1].item<float>();
  float box2_x2 = box2[2].item<float>();
  float box2_y2 = box2[3].item<float>();

  float x1 = std::max(box1_x1, box2_x1);
  float y1 = std::max(box1_y1, box2_y1);
  float x2 = std::min(box1_x2, box2_x2);
  float y2 = std::min(box1_y2, box2_y2);

  if (x1 >= x2 || y1 >= y2) {
    return 0.0f;
  }

  return (x2 - x1) * (y2 - y1);
}

float unioner(torch::Tensor box1, torch::Tensor box2) {
  float box1_x1 = box1[0].item<float>();
  float box1_y1 = box1[1].item<float>();
  float box1_x2 = box1[2].item<float>();
  float box1_y2 = box1[3].item<float>();

  float box2_x1 = box2[0].item<float>();
  float box2_y1 = box2[1].item<float>();
  float box2_x2 = box2[2].item<float>();
  float box2_y2 = box2[3].item<float>();

  float box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
  float box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);

  return box1_area + box2_area - intersection(box1, box2);
}

float iou(torch::Tensor box1, torch::Tensor box2) {
  return intersection(box1, box2) / unioner(box1, box2);
}

int main() {
  std::vector<std::string> yolo_classes = {<classes name "","","">};

  cv::Mat img;
  img = cv::imread(<image-path>);
  cv::Size size = img.size();
  int img_width = size.width;
  int img_height = size.height;
  
  torch::Tensor input = load_input_image(<image-path>, <image-size>);
  
  torch::jit::script::Module module = torch::jit::load(<torchscript-mode-path>);
   
  auto start_time = std::chrono::high_resolution_clock::now();
  torch::Tensor output = module.forward({input}).toTensor();
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time); 
  std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
  output = output[0].transpose(0,1);
  torch::Tensor row = output[0];
  std::cout << output.sizes() << std::endl;

  

  std::vector<std::vector<float>> filteredRows;
  
  for(int i=0; i<output.size(0); i++) {
    
    torch::Tensor row = output[i];
    
    std::vector<float> parsedRow = parse_row(row, img_width, img_height);
    
    if(parsedRow[4] > 0.7) {
      filteredRows.push_back(parsedRow);
    }

  }

  std::cout << filteredRows << std::endl;
  for(auto& row : filteredRows) {

    for(int i=0; i<row.size(); i+=5) {

      float x1 = row[i];
      float y1 = row[i+1]; 
      float x2 = row[i+2];
      float y2 = row[i+3];
      //auto  label = std::to_string(row[i+4]);

      cv::rectangle(img, cv::Point(x1,y1), cv::Point(x2,y2),  
                     cv::Scalar(0,255,0), 2);

     // cv::putText(img, label, cv::Point(x1,y1), 
       //           cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0));

    }

  }
  

  cv::imshow("Image", img);

  cv::waitKey(0);

}

