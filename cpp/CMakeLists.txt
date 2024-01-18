cmake_minimum_required(VERSION 3.2)

project(mobese)

# Torch'un bulunduğu dizin
set(CMAKE_PREFIX_PATH /home/cha0/Downloads/libtorch)

# OpenCV bul ve projeye ekle
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Torch bul ve projeye ekle
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

include_directories(/home/cha0/Desktop/cpp/ktscpp)
# Derleme seçeneklerini ayarla
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Ana uygulamayı tanımla
add_executable(${PROJECT_NAME} main.cpp)

# Bağımlılıkları ve standardı ayarla
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

