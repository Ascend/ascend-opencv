# Opencv ACL模块安装及使用<a name="ZH-CN_TOPIC_0302083215"></a>

## 功能描述<a name="section1421916179418"></a>

该模块实现了Opencv部分模块对AscendCL的支持,包括MAT类及部分矩阵操作函数，具体见API接口文档



## 目录结构<a name="section8733528154320"></a>

```
├── CMakeLists.txt                    //Cmake配置
├── include                           //头文件目录
│   └── opencv2
│       └── acl
│           ├── acl.hpp               //ACL头文件
│           ├── acl_init.hpp          //ACL初始化模块类的声明
│           ├── acl_mat.hpp           //aclMat类的声明
│           ├── acl_type.hpp          //ACL类型声明
│           ├── gemm.hpp              //gemm模块
│           ├── init_core.hpp         //ACL初始化环境核心实现
│           ├── mat_core.hpp          //Mat类核心实现
│           ├── mathfuncs.hpp         //math函数模块
│           ├── matrices.hpp          //矩阵操作模块
│           └── operator_desc.hpp     //算子描述模块
├── README_CN.md
├── run.sh                            //自动化部署脚本
├── src                               //源文件目录,对应声明
│   ├── acl_init.cpp
│   ├── acl_mat.cpp
│   ├── gemm.cpp
│   ├── mathfuncs.cpp
│   ├── matrices.cpp
│   ├── operator_desc.cpp
│   └── precomp.hpp                   //头文件总包含
└── test                              //单元测试目录
    ├── acl.cpp                       //总测试模块
    ├── acl.json
    ├── test_acl.cpp                  //aclMat类重载运算符测试
    ├── test_common.cpp               //测试公用模块
    ├── test_common.hpp               //测试公用模块声明
    ├── test_correctness.cpp          //函数正确性验证
    ├── test_correctness.hpp
    ├── test_gemm.cpp                 //gemm模块性能验证
    ├── test_main.cpp         
    ├── test_mathfuncs.cpp            //math函数模块性能验证
    ├── test_matrices.cpp             //矩阵操作模块性能验证
    ├── test_perf.hpp
    └── test_precomp.hpp              //测试头文件总包含
```

## 环境要求<a name="zh-cn_topic_0230709958_section1256019267915"></a>

-   操作系统及架构：CentOS x86\_64、CentOS aarch64、Ubuntu 18.04 x86\_64、EulerOS x86、EulerOS aarch64
-   编译器：
    -   运行环境操作系统架构为x86时，编译器为g++
    -   运行环境操作系统架构为arm64时，编译器为aarch64-linux-gnu-g++
-   python及依赖的库：Python3.7.*x*（3.7.0 ~ 3.7.11）、Python3.8.*x*（3.8.0 ~ 3.8.11）
-   已完成昇腾AI软件栈的部署。


## 配置环境变量

- 开发环境上环境变量配置

  1. CANN-Toolkit包提供进程级环境变量配置脚本，供用户在进程中引用，以自动完成CANN基础环境变量的配置，配置示例如下所示

     ```
     . ${HOME}/Ascend/ascend-toolkit/set_env.sh
     ```

     “$HOME/Ascend”请替换“Ascend-cann-toolkit”包的实际安装路径。

  2. 算子编译依赖Python，以Python3.7.5为例，请以运行用户执行如下命令设置Python3.7.5的相关环境变量。

     ```
     #用于设置python3.7.5库文件路径
     export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
     #如果用户环境存在多个python3版本，则指定使用python3.7.5版本
     export PATH=/usr/local/python3.7.5/bin:$PATH
     ```

     Python3.7.5安装路径请根据实际情况进行替换，您也可以将以上命令写入~/.bashrc文件中，然后执行source ~/.bashrc命令使其立即生效。

  3. 开发环境上，设置环境变量，配置AscendCL单算子验证程序编译依赖的头文件与库文件路径。

     编译脚本会按环境变量指向的路径查找编译依赖的头文件和库文件，“$HOME/Ascend”请替换“Ascend-cann-toolkit”包的实际安装路径。

     - 当运行环境操作系统架构是x86时，配置示例如下所示：

       ```
       export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest/x86_64-linux
       export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub
       ```

     - 当运行环境操作系统架构时AArch64时，配置示例如下所示：

       ```
       export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest/arm64-linux
       export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub
       ```

- 运行环境上环境变量配置

  - 若运行环境上安装的“Ascend-cann-toolkit”包，环境变量设置如下：

    ```
    . ${HOME}/Ascend/ascend-toolkit/set_env.sh
    ```

  - 若运行环境上安装的“Ascend-cann-nnrt”包，环境变量设置如下：

    ```
    . ${HOME}/Ascend/nnrt/set_env.sh
    ```

  - 若运行环境上安装的“Ascend-cann-nnae”包，环境变量设置如下：

    ```
    . ${HOME}/Ascend/nnae/set_env.sh
    ```

    “$HOME/Ascend”请替换相关软件包的实际安装路径。




## 安装说明
1. 在配置好AScend之后,用户需要官网下载好opencv库,并将本模块(acl模块)和opencv在同一级目录下
2. 运行命令:cd acl && mv run.sh ../,确保acl模块、opencv、run.sh在同一目录下
3. 如果AScend安装的路径不是系统默认路径，修改acl/CMakelists.txt文件中acl_lib，acl_inc路径
5. 如果需要运行测试案例需要修改test目录下acl.cpp中的set_device函数中.json文件的路径为绝对路径
6. 给脚本文件加权限: chmod +x run.sh
7. 单独运行安装脚本: ./run.sh，如果系统架构为x86，运行：./run.sh -x86
8. 运行安装并且启动单元测试模块: ./run.sh ACLTEST，如果系统架构为x86，运行：./run.sh -x86 ACLTEST


## 单独测试步骤说明
1、acl库安装成功之后，进入opencv/build/bin目录下
2、找到生成的相对应的测试可执行文件opencv_test_acl
3、测试全部模块可以直接运行opencv_test_acl
4、如果要测试单独某个模块，参照acl/test/acl.cpp里面TEST函数
5、例如：TEST(ACLMAT_CONSTRUCTOR, MEMORY_ALIGN)、TEST(Gemm, MatMul)，可以采用命令： ./opencv_test_acl --gtest_filter=[测试模块名字],如： ./opencv_test_acl --gtest_fliter=ACLMAT_CONSTRUCTOR.MEMORY_ALIGN
./opencv_test_acl --gtest_fliter=Gemm.MatMul



