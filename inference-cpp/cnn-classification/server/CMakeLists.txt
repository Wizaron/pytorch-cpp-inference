cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cnn-classification)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(cnn-inference main.cc base64.cc ../infer.cc ../../../utils/opencvutils.cc ../../../utils/torchutils.cc)

target_link_libraries(cnn-inference "${TORCH_LIBRARIES}")
target_link_libraries(cnn-inference "${OpenCV_LIBS}")
target_link_libraries(cnn-inference ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

set_property(TARGET cnn-inference PROPERTY CXX_STANDARD 11)
set_property(TARGET cnn-inference PROPERTY OUTPUT_NAME predict)
