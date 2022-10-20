# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(opencv_example_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
#find_package(OpenCV REQUIRED)

set(acl_lib "/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub/")
set(acl_lib "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/stub/")
link_directories(${acl_lib})

set(acl_inc "/usr/local/Ascend/ascend-toolkit/latest/acllib/include/")
set(acl_inc "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include/")
include_directories(${acl_inc})

set(cv_inc "/home/perfxlab4/include/opencv4")
include_directories(${cv_inc})

set(cv_lib "/home/perfxlab4/lib")
link_directories(${cv_lib})

set(cv_libs "opencv_core" "opencv_acl" "opencv_imgproc" "opencv_highgui" "opencv_imgcodecs")

set(ACL_LIBRARIES "ascendcl" "stdc++" "acl_op_compiler")

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(opencv_example example.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencv_example PRIVATE ${OpenCV_LIBS} ${ACL_LIBRARIES} ${cv_libs})
