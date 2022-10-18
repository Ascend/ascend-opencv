# Opencv ACL模块简单使用示例<a name="ZH-CN_TOPIC_0302083215"></a>

## 功能描述<a name="section1421916179418"></a>

主要演示了aclMat类的简单使用,用acl模块中的merge和split算子对图片进行操作

## 步骤说明
1. 在acl模块安装完成后，修改samples下CMakeLists.txt文件中Ascend安装路径(acl_inc，acl_lib)，opencv安装路径(cv_inc，cv_lib)
2. 在samples目录下创建build目录: mkdir build 
3. 进入build目录: cd build
4. 运行cmake: cmake ..
5. 编译: make 
6. 运行可执行文件opencv_example



