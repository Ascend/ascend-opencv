# Opencv ACL模块简单使用示例<a name="ZH-CN_TOPIC_0302083215"></a>

## 功能描述<a name="section1421916179418"></a>

主要演示了aclMat类的简单使用,用acl模块中的merge和split算子对图片进行操作

## 步骤说明
1. 在acl模块编译完成后，进入opencv下的build目录下：cd opencv/build
2. 如果没有权限在系统路径下安装opencv，需要自定义安装路径：cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/home/perfxlab4/test ..
3. 更改2步骤中/home/perfxlab4/test为想要安装的路径，再运行：sudo make install
4. 修改samples下CMakeLists.txt文件中Ascend安装路径(acl_inc，acl_lib)，opencv安装路径(cv_inc，cv_lib，若自定义，则根据上两步填写相应路径)
5. 在samples目录下创建build目录: mkdir build 
6. 进入build目录: cd build
7. 运行cmake: cmake ..
8. 编译: make 
9. 运行可执行文件opencv_example



