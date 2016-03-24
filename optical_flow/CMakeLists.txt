cmake_minimum_required(VERSION 2.8)
PROJECT(Optical_Flow_TEST)

set(CMAKE_BUILD_TYPE Debug)

set(OPENCV_310 /home/christsaga/opencv-3.1.0/opencv/release)
set(OPENCV_310_INCLUDE ${OPENCV_310}/include)

include_directories( ${OPENCV_310_INCLUDE} )

find_library(OPENCV_310_CORE
                NAMES libopencv_core.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_CALIB3D
                NAMES libopencv_calib3d.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_FEATURE2D
                NAMES libopencv_features2d.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_FLANN
                NAMES libopencv_flann.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_IMGPROC
                NAMES libopencv_imgproc.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_IMGCODECS
                NAMES libopencv_imgcodecs.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_VIDEO
                NAMES libopencv_video.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_VIDEOIO
                NAMES libopencv_videoio.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_HIGHGUI
                NAMES libopencv_highgui.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)

find_library(OPENCV_310_XFEATURE2D
                NAMES libopencv_xfeatures2d.so
                PATHS ${OPENCV_310}/lib
                NO_DEFAULT_PATH)



set(OPENCV_310_LIB ${OPENCV_310_CORE}
                    ${OPENCV_310_CALIB3D}
                    ${OPENCV_310_FEATURE2D}
                    ${OPENCV_310_FLANN}
                    ${OPENCV_310_IMGPROC}
                    ${OPENCV_310_IMGCODECS}
                    ${OPENCV_310_VIDEO}
                    ${OPENCV_310_VIDEOIO}
                    ${OPENCV_310_HIGHGUI}
                    ${OPENCV_310_XFEATURE2D})

set(sample_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/)


add_executable( OF_basic_test ${sample_dir}main.cpp)

target_link_libraries(OF_basic_test ${OPENCV_310_LIB})
