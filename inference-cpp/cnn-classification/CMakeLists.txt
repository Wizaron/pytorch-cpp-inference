cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cnn-classification)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(cnn-inference infer.cc ../../utils/opencvutils.cc ../../utils/torchutils.cc)

target_link_libraries(cnn-inference "${TORCH_LIBRARIES}")
target_link_libraries(cnn-inference "${OpenCV_LIBS}")

set_property(TARGET cnn-inference PROPERTY CXX_STANDARD 11)
set_property(TARGET cnn-inference PROPERTY OUTPUT_NAME predict)
