#include "test_common.hpp"
#include "test_perf.hpp"

#define CHECK(cmd) do {                        \
  aclError e = cmd;                            \
  if( e != ACL_ERROR_NONE) {                   \
    printf("Failed: ACL error %s:%d '%d'\n",   \
        __FILE__,__LINE__,e);                  \
    exit(0);                                   \
  }                                            \
} while(0)

void PERF_TEST::Test_operator_add_perf(aclCxt *acl_context)
{
    int val;
    int valmax = 8192;
    double begin, end, time, acltime;
    Common_Test test;

    vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1, CV_64FC1};
    for (size_t i = 0; i < type.size(); ++i)
    {
        test.PrintLog("Perf test : Function: operator+=()", type[i]);
        for (val = 8; val <= valmax; val *= 2)
        {
            int n = 100;
            Mat mat_src(val, val, type[i]);
            Mat mat_dest(val, val, type[i]);
            Mat mat_dest1(val, val, type[i]);

            test.SetDataRange(mat_src, 1);
            test.SetDataRange(mat_dest, 1);

            aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
            aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

            begin = static_cast<double>(getTickCount());
            while (n--)
                mat_dest += mat_src;
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency();

            n = 100;
            begin = static_cast<double>(getTickCount());
            while (n--)
                aclmat_dest += aclmat_src;
            end = static_cast<double>(getTickCount());
            acltime = (end - begin) / getTickFrequency();

            aclmat_dest.download(mat_dest1);
            bool ret = test.Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
            if (val < 128)
                cout << "Shape: " << val << " x " << val << "\t\t";
            else
                cout << "Shape: " << val << " x " << val << "\t";
            cout << "CpuTimes: " << time << "\tAclTimes: " << acltime << "\tRate: " << time / acltime << endl;
        }
    }
}

void PERF_TEST::Test_operator_sub_perf(aclCxt *acl_context)
{
    int val;
    int valmax = 8192;
    double begin, end, time, acltime;
    Common_Test test;

    vector<int> type{CV_32FC1, CV_32SC1, CV_64FC1};
    for (size_t i = 0; i < type.size(); ++i)
    {
        test.PrintLog("Perf test : Function: operator-=()", type[i]);
        for (val = 8; val <= valmax; val *= 2)
        {
            int n = 100;
            Mat mat_src(val, val, type[i]);
            Mat mat_dest(val, val, type[i]);
            Mat mat_dest1(val, val, type[i]);

            test.SetDataRange(mat_src, 4);
            test.SetDataRange(mat_dest, 32);

            aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
            aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

            begin = static_cast<double>(getTickCount());
            while (n--)
                mat_dest -= mat_src;
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency();

            n = 100;
            begin = static_cast<double>(getTickCount());
            while (n--)
                aclmat_dest -= aclmat_src;
            end = static_cast<double>(getTickCount());
            acltime = (end - begin) / getTickFrequency();

            aclmat_dest.download(mat_dest1);
            bool ret = test.Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
            if (val < 128)
                cout << "Shape: " << val << " x " << val << "\t\t";
            else
                cout << "Shape: " << val << " x " << val << "\t";
            cout << "CpuTimes: " << time << "\tAclTimes: " << acltime << "\tRate: " << time / acltime << endl;
        }
    }
    
}

void PERF_TEST::Test_operator_div_perf(aclCxt *acl_context)
{
    int val;
    int valmax = 8192;
    double begin, end, time, acltime;
    Common_Test test;

    vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1, CV_64FC1};
    for (size_t i = 0; i < type.size(); ++i)
    {
        test.PrintLog("Perf test : Function: operator/=()", type[i]);
        for (val = 8; val <= valmax; val *= 2)
        {
            int n = 100;
            Mat mat_src(val, val, type[i], Scalar(1, 2, 4));
            Mat mat_dest(val, val, type[i], Scalar(2, 4, 8));
            Mat mat_dest1(val, val, type[i]);

            aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
            aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

            begin = static_cast<double>(getTickCount());
            while (n--)
                mat_dest /= mat_src;
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency();

            n = 100;
            begin = static_cast<double>(getTickCount());
            while (n--)
                aclmat_dest /= aclmat_src;
            end = static_cast<double>(getTickCount());
            acltime = (end - begin) / getTickFrequency();

            aclmat_dest.download(mat_dest1);
            bool ret = test.Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
            if (val < 128)
                cout << "Shape: " << val << " x " << val << "\t\t";
            else
                cout << "Shape: " << val << " x " << val << "\t";
            cout << "CpuTimes: " << time << "\tAclTimes: " << acltime << "\tRate: " << time / acltime << endl;
        }
    }
    
}

void PERF_TEST::Test_operator_mul_perf(aclCxt *acl_context)
{
    int val, type;
    int valmax = 4096;
    double begin, end, time, acltime;
    Common_Test test;

    type = CV_32FC1;
    for (val = 8; val <= valmax; val *= 2)
    {
        int n = 100;
        Mat mat_src(val, val, type);
        Mat mat_dest(val, val, type);
        Mat mat_dest1(val, val, type);

        test.SetDataRange(mat_src, 1); 
        test.SetDataRange(mat_dest, 1); 

        aclMat aclmat_src(val, val, type, mat_src.data, acl_context);
        aclMat aclmat_dest(val, val, type, mat_dest.data, acl_context);

        begin = static_cast<double>(getTickCount());
        while (n--)
            mat_dest *= mat_src;
        end = static_cast<double>(getTickCount());
        time = (end - begin) / getTickFrequency();

        n = 100;
        begin = static_cast<double>(getTickCount());
        while (n--)
            aclmat_dest *= aclmat_src;
        end = static_cast<double>(getTickCount());
        acltime = (end - begin) / getTickFrequency();

        aclmat_dest.download(mat_dest1);
        bool ret = test.Test_Diff(mat_dest, mat_dest1);
        ASSERT_TRUE(ret);
        if (val < 128)
            cout << "Shape: " << val << " x " << val << "\t\t";
        else
            cout << "Shape: " << val << " x " << val << "\t";
        cout << "CpuTimes: " << time << "\tAclTimes: " << acltime << "\tRate: " << time / acltime << endl;
    }
}

void PERF_TEST::Test_other(aclCxt *acl_context)
{
    std::vector<aclDataBuffer *> input_buffers_;
    std::vector<aclDataBuffer *> output_buffers_;
    std::vector<aclTensorDesc *> input_descs_;
    std::vector<aclTensorDesc *> output_descs_;

    string op_type_ = "ConcatD";
    auto *attr_ = aclopCreateAttr();
    vector<int64_t> a = {0};
    aclopSetAttrInt(attr_, "N", 2);
    aclopSetAttrInt(attr_, "concat_dim", 0);

    vector<int64_t> dims0 = {2, 4};
    auto size0 = 2 * 4 * 4;
    auto *desc0 = aclCreateTensorDesc(ACL_FLOAT, dims0.size(), dims0.data(), ACL_FORMAT_NCHW);
    void *ptr0;
    vector<float> data0;
    for (auto i = 0; i < 8; ++i)
    {
        data0.emplace_back(i);
    }
    CHECK(aclrtMalloc(&ptr0, 2 * 4 * 4, ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtMemcpy(ptr0, data0.size() * 4, data0.data(), data0.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    auto *buffer0 = aclCreateDataBuffer(ptr0, size0);
    input_descs_.push_back(desc0);
    input_buffers_.push_back(buffer0);
    cout << "input0 done" << endl;

    vector<int64_t> dims1 = {2, 4};
    auto *desc1 = aclCreateTensorDesc(ACL_FLOAT, dims1.size(), dims1.data(), ACL_FORMAT_NCHW);
    input_descs_.push_back(desc1);
    void *ptr1;
    CHECK(aclrtMalloc(&ptr1, 1024, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<float> data1;
    for (auto i = 0; i < 8; ++i)
    {
        data1.emplace_back(i);
    }
    aclrtMemcpy(ptr1, data1.size() * 4, data1.data(), data1.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    auto *buffer1 = aclCreateDataBuffer(ptr1, 2 * 4 * 4);
    input_buffers_.push_back(buffer1);
    cout << "input1 done" << endl;

    vector<int64_t> dims2 = {4, 4};
    auto *desc2 = aclCreateTensorDesc(ACL_FLOAT, dims2.size(), dims2.data(), ACL_FORMAT_NCHW);
    output_descs_.push_back(desc2);
    void *ptr2;
    CHECK(aclrtMalloc(&ptr2, 1024, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<float> data2;
    for (auto i = 0; i < 256; ++i)
    {
        data1.emplace_back(i);
    }
    aclrtMemcpy(ptr2, data2.size() * 4, data2.data(), data2.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    auto *buffer2 = aclCreateDataBuffer(ptr2, 4 * 4 * 4);
    output_buffers_.push_back(buffer2);
    cout << "output0 done" << endl;

    aclError ret = aclopCompileAndExecute(
        op_type_.c_str(), input_descs_.size(), input_descs_.data(),
        input_buffers_.data(), output_descs_.size(), output_descs_.data(),
        output_buffers_.data(), attr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL,
        acl_context->get_stream(0));

    std::cout << "aclopCompileAndExecutr:" << ret << std::endl;
    CHECK(aclrtSynchronizeStream(acl_context->get_stream(0)));

    std::cout << "aclrtSynchronizeStream ok" << std::endl;
    vector<float> res;
    for (auto i = 0; i < 256 + 256; ++i)
    {
        res.emplace_back(i);
    }
    CHECK(aclrtMemcpy(res.data(), res.size() * 4, ptr2, res.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));

    for (auto item : res)
    {
        cout << item << " ";
    }
    cout << endl;
}

void PERF_TEST::Test_other1(aclCxt *acl_context)
{
    std::vector<aclDataBuffer *> input_buffers_;
    std::vector<aclDataBuffer *> output_buffers_;
    std::vector<aclTensorDesc *> input_descs_;
    std::vector<aclTensorDesc *> output_descs_;

    string op_type_ = "ConcatD";
    auto *attr_ = aclopCreateAttr();
    vector<int64_t> a = {0};
    aclopSetAttrInt(attr_, "N", 2);
    aclopSetAttrInt(attr_, "concat_dim", 0);

    Common_Test test;
    Mat src(2, 4, CV_32FC1); 
    test.SetDataRange(src, 8); 
    aclMat acl_src(2, 4, CV_32FC1, src.data, acl_context);
    vector<int64_t> dims0 = {2, 4};
    auto size0 = 2 * 4 * 4;
    auto *desc0 = aclCreateTensorDesc(ACL_FLOAT, dims0.size(), dims0.data(), ACL_FORMAT_NHWC);

    auto *buffer0 = aclCreateDataBuffer(acl_src.data, size0);
    input_descs_.push_back(desc0);
    input_buffers_.push_back(buffer0);
    std::cout << "input0 done" << endl;

    Mat src1(2, 4, CV_32FC1); 
    test.SetDataRange(src1, 8); 
    aclMat acl_src1(2, 4, CV_32FC1, src1.data, acl_context);
    vector<int64_t> dims1 = {2, 4};
    auto size1 = 2 * 4 * 4;
    auto *desc1 = aclCreateTensorDesc(ACL_FLOAT, dims1.size(), dims1.data(), ACL_FORMAT_NHWC);

    auto *buffer1 = aclCreateDataBuffer(acl_src1.data, size1);
    input_descs_.push_back(desc1);
    input_buffers_.push_back(buffer1);
    std::cout << "input1 done" << endl;

    aclMat acl_dest(4, 4, CV_32FC1, acl_context);
    vector<int64_t> dims2 = {4, 4};
    auto size3 = 4 * 4 * 4;
    auto *desc2 = aclCreateTensorDesc(ACL_FLOAT, dims2.size(), dims2.data(), ACL_FORMAT_NHWC);

    auto *buffer2 = aclCreateDataBuffer(acl_dest.data, size3);
    output_descs_.push_back(desc2);
    output_buffers_.push_back(buffer2);
    std::cout << "output0 done" << endl;

    aclError ret = aclopCompileAndExecute(
        op_type_.c_str(), input_descs_.size(), input_descs_.data(),
        input_buffers_.data(), output_descs_.size(), output_descs_.data(),
        output_buffers_.data(), attr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL,
        acl_context->get_stream(0));

    std::cout << "aclopCompileAndExecutr:" << ret << std::endl;
    CHECK(aclrtSynchronizeStream(acl_context->get_stream(0)));

    std::cout << "aclrtSynchronizeStream ok" << std::endl;
    vector<float> res;
    for (auto i = 0; i < 256 + 256; ++i)
    {
        res.emplace_back(i);
    }
    CHECK(aclrtMemcpy(res.data(), res.size() * 4, acl_dest.data, res.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));

    for (auto item : res)
    {
        std::cout << item << " ";
    }
    std::cout << endl;
}

void PERF_TEST::Test_other2()
{
    CHECK(aclInit(nullptr));
    std::cout << "aclInit ok" << std::endl;

    CHECK(aclrtSetDevice(0));
    std::cout << "aclrtSetDevice 0 ok" << std::endl;

    std::vector<aclDataBuffer *> input_buffers_;
    std::vector<aclDataBuffer *> output_buffers_;
    std::vector<aclTensorDesc *> input_descs_;
    std::vector<aclTensorDesc *> output_descs_;

    string op_type_ = "ConcatD";
    auto *attr_ = aclopCreateAttr();
    vector<int64_t> a = {0};
    aclopSetAttrInt(attr_, "N", 2);
    aclopSetAttrInt(attr_, "concat_dim", 0);

    vector<int64_t> dims0 = {2, 4};
    auto size0 = 2 * 4 * 4;
    auto *desc0 = aclCreateTensorDesc(ACL_FLOAT, dims0.size(), dims0.data(), ACL_FORMAT_NCHW);
    void *ptr0;
    vector<float> data0;
    for (auto i = 0; i < 8; ++i)
    {
        data0.emplace_back(i);
    }
    CHECK(aclrtMalloc(&ptr0, 2 * 4 * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    //  std::cout << "ptr:" << ptr0 << " ptr+256:" << ptr0+256;

    aclrtMemcpy(ptr0, data0.size() * 4, data0.data(), data0.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    auto *buffer0 = aclCreateDataBuffer(ptr0, size0);
    input_descs_.push_back(desc0);
    input_buffers_.push_back(buffer0);
    cout << "input0 done" << endl;

    vector<int64_t> dims1 = {2, 4};
    auto *desc1 = aclCreateTensorDesc(ACL_FLOAT, dims1.size(), dims1.data(), ACL_FORMAT_NCHW);
    input_descs_.push_back(desc1);
    void *ptr1;
    CHECK(aclrtMalloc(&ptr1, 1024, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<float> data1;
    for (auto i = 0; i < 8; ++i)
    {
        data1.emplace_back(i);
    }
    aclrtMemcpy(ptr1, data1.size() * 4, data1.data(), data1.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    auto *buffer1 = aclCreateDataBuffer(ptr1, 2 * 4 * 4);
    input_buffers_.push_back(buffer1);
    cout << "input1 done" << endl;

    vector<int64_t> dims2 = {4, 4};
    auto *desc2 = aclCreateTensorDesc(ACL_FLOAT, dims2.size(), dims2.data(), ACL_FORMAT_NCHW);
    output_descs_.push_back(desc2);
    void *ptr2;
    CHECK(aclrtMalloc(&ptr2, 1024, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<float> data2;
    for (auto i = 0; i < 256; ++i)
    {
        data1.emplace_back(i);
    }
    aclrtMemcpy(ptr2, data2.size() * 4, data2.data(), data2.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    auto *buffer2 = aclCreateDataBuffer(ptr2, 4 * 4 * 4);
    output_buffers_.push_back(buffer2);
    cout << "output0 done" << endl;

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    cout << 2 << endl;
    aclError ret = aclopCompileAndExecute(
        op_type_.c_str(), input_descs_.size(), input_descs_.data(),
        input_buffers_.data(), output_descs_.size(), output_descs_.data(),
        output_buffers_.data(), attr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL,
        stream);

    cout << 3 << endl;
    std::cout << "aclopCompileAndExecutr:" << ret << std::endl;
    CHECK(aclrtSynchronizeStream(stream));

    std::cout << "aclrtSynchronizeStream ok" << std::endl;
    vector<float> res;
    for (auto i = 0; i < 256 + 256; ++i)
    {
        res.emplace_back(i);
    }
    CHECK(aclrtMemcpy(res.data(), res.size() * 4, ptr2, res.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));

    for (auto item : res)
    {
        cout << item << " ";
    }
    cout << endl;
}