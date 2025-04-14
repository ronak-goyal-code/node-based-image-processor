cmake_minimum_required(VERSION 3.10)
project(NodeBasedImageProcessor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

#-----------------------------------------------------------------------------

set(Qt5_DIR "C:/Qt/5.15.2/msvc2017_64/lib/cmake/Qt5")
set(OpenCV_DIR "C:/tools/opencv/build")

find_package(Qt5 COMPONENTS Widgets Concurrent REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} Qt5::Widgets Qt5::Concurrent ${OpenCV_LIBS})
