#include "test_common.hpp"
#include "test_perf.hpp"

/*
//disable
void PERF_TEST::Test_Lookuptable(aclCxt *acl_context_0)
{
    int type = CV_8UC1;
    Common_Test test;
    Mat mat_src(1, 256, type);
    Mat mat_dest(1, 256, type);
    Mat lookuptable(1, 256, type);

    test.SetDataRange(mat_src, 32);
    test.SetDataRange(lookuptable, 32);

    aclMat aclmat_src(1, 256, type, mat_src.data, acl_context_0);
    aclMat aclmat_dest(1, 256, type, mat_dest.data, acl_context_0);
    aclMat lut(1, 256, type, lookuptable.data, acl_context_0);
    // LUT(mat_src, lookuptable, mat_dest);
    lookUpTable(aclmat_src, lut, aclmat_dest);
    cout << mat_src << endl;
    cout << lookuptable << endl;
    cout << mat_dest << endl;
}
*/


void PERF_TEST::Test_Merge(aclCxt *acl_context)
{
    int val;
    int valmax = 8192;
    double begin, end, time, acltime;
    Common_Test test;

    vector<int> srcType{CV_32FC1};
    vector<int> destType{CV_32FC3};

    for (size_t i = 0; i < srcType.size(); ++i)
    {
        test.PrintLog("Perf test : Function: merge()", srcType[i]);
        for (val = 8; val <= valmax; val *= 2)
        {
            int n = 100;
            Mat mat_src1(val, val, srcType[i], Scalar(1));
            Mat mat_src2(val, val, srcType[i], Scalar(2));
            Mat mat_src3(val, val, srcType[i], Scalar(3));
            Mat mat_dest(val, val, destType[i]);
            Mat mat_dest1(val, val, destType[i]);

            test.SetDataRange(mat_src1, 32);
            test.SetDataRange(mat_src2, 32);
            test.SetDataRange(mat_src3, 32);

            aclMat aclmat_src1(val, val, srcType[i], mat_src1.data, acl_context);
            aclMat aclmat_src2(val, val, srcType[i], mat_src2.data, acl_context);
            aclMat aclmat_src3(val, val, srcType[i], mat_src3.data, acl_context);
            aclMat aclmat_dest(val, val, destType[i], mat_dest.data, acl_context);

            vector<Mat> src;
            src.emplace_back(mat_src1);
            src.emplace_back(mat_src2);
            src.emplace_back(mat_src3);

            vector<aclMat> acl_src;
            acl_src.emplace_back(aclmat_src1);
            acl_src.emplace_back(aclmat_src2);
            acl_src.emplace_back(aclmat_src3);

            begin = static_cast<double>(getTickCount());
            while (n--)
                merge(src, mat_dest);
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency();

            n = 100;
            begin = static_cast<double>(getTickCount());
            while (n--)
                merge(acl_src, aclmat_dest);
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


void PERF_TEST::Test_Transpose(aclCxt *acl_context)
{
    int val;
    int valmax = 8192;
    double begin, end, time, acltime;
    Common_Test test;

    vector<int> type{CV_32FC1, CV_32SC1};
    for (size_t i = 0; i < type.size(); ++i)
    {
        test.PrintLog("Perf test : Function: transpose()", type[i]);
        for (val = 8; val <= valmax; val *= 2)
        {
            int n = 100;
            Mat mat_src(val, val, type[i]);
            Mat mat_dest(val, val, type[i]);
            Mat mat_dest1(val, val, type[i]);

            test.SetDataRange(mat_src, 32);

            aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
            aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

            begin = static_cast<double>(getTickCount());
            while (n--)
                transpose(mat_src, mat_dest);
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency();

            n = 100;
            begin = static_cast<double>(getTickCount());
            while (n--)
                transpose(aclmat_src, aclmat_dest);
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

void PERF_TEST::Test_Split(aclCxt *acl_context)
{
    int val;
    int valmax = 8;
    double begin, end, time, acltime;
    Common_Test test;

    vector<int> srcType{CV_32FC3};
    vector<int> destType{CV_32FC1};

    for (size_t i = 0; i < srcType.size(); ++i)
    {
        test.PrintLog("Perf test : Function: split()", srcType[i]);
        for (val = 8; val <= valmax; val *= 2)
        {
            int n = 1;
            Mat mat_src(val, val, srcType[i]);
            Mat mat_dest1(val, val, destType[i]);
            Mat mat_dest2(val, val, destType[i]);
            Mat mat_dest3(val, val, destType[i]);

            test.SetDataRange(mat_src, 32);

            aclMat aclmat_src(val, val, srcType[i], mat_src.data, acl_context);
            aclMat aclmat_dest1(val, val, destType[i], mat_dest1.data, acl_context);
            aclMat aclmat_dest2(val, val, destType[i], mat_dest2.data, acl_context);
            aclMat aclmat_dest3(val, val, destType[i], mat_dest3.data, acl_context);

            vector<Mat> dest;
            dest.emplace_back(mat_dest1);
            dest.emplace_back(mat_dest2);
            dest.emplace_back(mat_dest3);

            vector<aclMat> acl_dest;
            acl_dest.emplace_back(aclmat_dest1);
            acl_dest.emplace_back(aclmat_dest2);
            acl_dest.emplace_back(aclmat_dest3);

            begin = static_cast<double>(getTickCount());
            while (n--)
                split(mat_src, dest);
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency();

            n = 1;
            begin = static_cast<double>(getTickCount());
            while (n--)
                split(aclmat_src, acl_dest);
            end = static_cast<double>(getTickCount());
            acltime = (end - begin) / getTickFrequency();

            (acl_dest.data())[0].download(mat_dest1);
            (acl_dest.data())[1].download(mat_dest2);
            (acl_dest.data())[2].download(mat_dest3);

            bool ret = test.Test_Diff((dest.data())[0], mat_dest1);
            ret &= test.Test_Diff((dest.data())[1], mat_dest2);
            ret &= test.Test_Diff((dest.data())[2], mat_dest3);
            ASSERT_TRUE(ret);
            if (val < 128)
                cout << "Shape: " << val << " x " << val << "\t\t";
            else
                cout << "Shape: " << val << " x " << val << "\t";
            cout << "CpuTimes: " << time << "\tAclTimes: " << acltime << "\tRate: " << time / acltime << endl;
        }
    }

}


void PERF_TEST::Test_Flip(aclCxt *acl_context)
{
    int val;
    int valmax = 8192;
    double begin, end, time, acltime;
    Common_Test test;

    vector<int> type{CV_8UC1, CV_32FC1, CV_32SC1, CV_64FC1};
    for (size_t i = 0; i < type.size(); ++i)
    {
        test.PrintLog("Perf test : Function: flip()", type[i]);
        for (val = 8; val <= valmax; val *= 2)
        {
            int n = 100;
            Mat mat_src(val, val, type[i]);
            Mat mat_dest(val, val, type[i]);
            Mat mat_dest1(val, val, type[i]);

            test.SetDataRange(mat_src, 32);

            aclMat aclmat_src(val, val, type[i], mat_src.data, acl_context);
            aclMat aclmat_dest(val, val, type[i], mat_dest.data, acl_context);

            begin = static_cast<double>(getTickCount());
            while (n--)
                flip(mat_src, mat_dest, 0);
            end = static_cast<double>(getTickCount());
            time = (end - begin) / getTickFrequency();

            n = 100;
            begin = static_cast<double>(getTickCount());
            while (n--)
                flip(aclmat_src, aclmat_dest, 0);
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