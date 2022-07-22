#if(NOT HAVE_ACL)
#  ocv_module_disable(acl)
#  return()
#endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" "-DENABLE_DVPP_INTERFACE")

set(acl_lib "/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub/")
set(acl_lib "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/stub/")
link_directories(${acl_lib})

set(acl_inc "/usr/local/Ascend/ascend-toolkit/latest/acllib/include/")
set(acl_inc "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include/")
ocv_include_directories(${acl_inc})

set(ASCEND_LIBRARIES "ascendcl" "stdc++" "acl_op_compiler")

set(the_description "ACL-accelerated Computer Vision")
ocv_define_module(acl opencv_core opencv_imgproc opencv_features2d opencv_objdetect opencv_video opencv_calib3d opencv_ml "${ASCEND_LIBRARIES}")

ocv_target_link_libraries(${the_module} "${ASCEND_LIBRARIES}")

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wshadow -Woverloaded-virtual -Wunused-private-field)
