#include "test_common.hpp"
#include "test_perf.hpp"

void PERF_TEST::Test_MatMul(aclCxt *acl_context)
{
    int val, n;
    int valmax = 4096;
    int cycle_index = 100;
    double begin, end, time, acltime;
    Common_Test test;
    vector<int> type{CV_32FC1};

    for (size_t i = 0; i < type.size(); ++i)
    {
        for (val = 8; val <= valmax; val *= 2)
        {
            Mat mat_src(val, val, type[i]);
            Mat mat_src1(val, val, type[i]);
            Mat mat_dest(val, val, type[i]);
            Mat mat_dest1(val, val, type[i]);

            test.SetDataRange(mat_src, 32);
            test.SetDataRange(mat_src1, 32);
            test.SetDataRange(mat_dest, 32);

            aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
            aclMat aclmat_src1(val, val, type[i], mat_src1.data, acl_context);
            aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

            n = cycle_index;
            begin = static_cast<double>(getTickCount());
            while (n--)
                mat_dest = mat_src * mat_src1;
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency() / cycle_index;

            n = (cycle_index - 1);
            MatMul(aclmat_src1, aclmat_src, aclmat_dest, 0);
            wait_stream(acl_context, 0);
            begin = static_cast<double>(getTickCount());
            while (n--)
                MatMul(aclmat_src1, aclmat_src, aclmat_dest, 1);
            wait_stream(acl_context, 1);
            end = static_cast<double>(getTickCount());
            acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

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

void PERF_TEST::Test_Convolution(aclCxt *acl_context)
{
    int val, n;
    int valmax = 4096;
    int cycle_index = 100;
    double begin, end, time, acltime;
    Common_Test test;
    vector<int> type{CV_32FC1};

    for (size_t i = 0; i < type.size(); ++i)
    {
        for (val = 8; val <= valmax; val *= 2)
        {
            Mat mat_src(val, val, type[i], Scalar{1, 2});
            Mat mat_kernel(3, 3, type[i], Scalar(1, 4));
            Mat mat_dest(val, val, type[i], Scalar{6});

            aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
            aclMat aclmat_kernel(3, 3, type[i], mat_kernel.data, acl_context);
            aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

            n = cycle_index;
            begin = static_cast<double>(getTickCount());
            while (n--)
                filter2D(mat_src, mat_dest, -1, mat_kernel);
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency() / cycle_index;

            vector<int64_t> strides{1, 1, 1, 1};
            vector<int64_t> pads{1, 1, 1, 1};
            n = (cycle_index - 1);
            Convolution(aclmat_src, aclmat_kernel, aclmat_dest, strides, pads, 0);
            wait_stream(acl_context, 0);
            begin = static_cast<double>(getTickCount());
            while (n--)
                Convolution(aclmat_src, aclmat_kernel, aclmat_dest, strides, pads, 1);
            wait_stream(acl_context, 1);
            end = static_cast<double>(getTickCount());
            Mat mat_dest1(aclmat_dest.rows, aclmat_dest.cols, type[i]);
            acltime = (end - begin) / getTickFrequency() / (cycle_index - 1);

            aclmat_dest.download(mat_dest1);
            /*
            bool ret = test.Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
            */
            if (val < 128)
                cout << "Shape: " << val << " x " << val << "\t\t";
            else
                cout << "Shape: " << val << " x " << val << "\t";
            cout << "CpuTimes: " << time << "\tAclTimes: " << acltime << "\tRate: " << time / acltime << endl;
        }
    }
}
