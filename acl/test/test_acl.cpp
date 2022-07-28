#include "test_common.hpp"
#include "test_perf.hpp"

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