#include "test_common.hpp"

Common_Test::Common_Test() {
    srand((unsigned)time(NULL));
}

Common_Test::~Common_Test() {

}

bool Common_Test::Test_Diff(const aclMat& aclmat, const Mat& mat, ALIGNMENT config) {
    bool is_correct;

    if (config == ALIGNMENT::MEMORY_UNALIGNED)
    {
        is_correct = (aclmat.rows == mat.rows);
        is_correct &= (aclmat.cols == mat.cols);
        is_correct &= (aclmat.channels() == mat.channels());
        is_correct &= (aclmat.type() == mat.type());
        is_correct &= (aclmat.step == mat.step);
        is_correct &= (aclmat.elemSize() == mat.elemSize());
        is_correct &= (aclmat.totalSize == mat.total() * mat.elemSize());
        is_correct &= ((aclmat.dataend - aclmat.datastart) == (mat.dataend - mat.datastart));
        
        Mat mat_dest(mat.rows, mat.cols, mat.type());
        aclmat.download(mat_dest);
        is_correct &= Test_Diff(mat, mat_dest);
    }
    else
    {
        is_correct = (aclmat.rows == mat.rows);
        is_correct &= (aclmat.cols == mat.cols);
        is_correct &= (aclmat.channels() == mat.channels());
        is_correct &= (aclmat.type() == mat.type());
        is_correct &= (aclmat.elemSize() == mat.elemSize());

        Mat mat_dest(mat.rows, mat.cols, mat.type());
        aclmat.download(mat_dest, MEMORY_ALIGN);
        is_correct &= Test_Diff(mat, mat_dest);
    }
    
    return is_correct;
}
    
bool Common_Test::Test_Diff(const aclMat& aclmat, const aclMat& aclmat_other) {
    bool is_correct;

    is_correct = (aclmat.flags == aclmat_other.flags);
    is_correct &= (aclmat.rows == aclmat_other.rows);
    is_correct &= (aclmat.cols == aclmat_other.cols);
    is_correct &= (aclmat.type() == aclmat_other.type());
    is_correct &= (aclmat.step == aclmat_other.step);
    is_correct &= (aclmat.data == aclmat_other.data);
    is_correct &= (aclmat.refcount == aclmat_other.refcount);
    is_correct &= (aclmat.datastart == aclmat_other.datastart);
    is_correct &= (aclmat.dataend == aclmat_other.dataend);
    is_correct &= (aclmat.offset == aclmat_other.offset);
    is_correct &= (aclmat.wholerows == aclmat_other.wholerows);
    is_correct &= (aclmat.wholecols == aclmat_other.wholecols);
    is_correct &= (aclmat.acl_context == aclmat_other.acl_context);
    is_correct &= (aclmat.totalSize == aclmat_other.totalSize);

    return is_correct;
}

bool Common_Test::Test_Diff(const Mat &mat, const Mat &mat_other)
{
    bool is_correct;
    
    is_correct = (mat.rows == mat_other.rows);
    is_correct &= (mat.cols == mat_other.cols);
    is_correct &= (mat.type() == mat_other.type());
    is_correct &= (mat.channels() == mat.channels());
    is_correct &= (mat.step == mat_other.step);
    is_correct &= (mat.elemSize() == mat_other.elemSize());
    is_correct &= (mat.total() == mat_other.total());

    switch (mat.depth()) 
    {
    case CV_8U:
        for (int i = 0; (is_correct == true) && (i < mat.rows * mat.cols * mat.channels()); i += mat.channels())
        {
            for (int j = 0; j < mat.channels(); ++j)
                is_correct &= ((mat.data)[i+j] == (mat_other.data)[i+j]);
        }
    return is_correct;
    case CV_16U:
        for (int i = 0; (is_correct == true) && (i < mat.rows * mat.cols * mat.channels()); i += mat.channels())
        {
            for (int j = 0; j < mat.channels(); ++j)
                is_correct &= (((unsigned short *)mat.data)[i+j] == ((unsigned short *)mat_other.data)[i+j]);
        }
    return is_correct;
    case CV_32S:
        for (int i = 0; (is_correct == true) && (i < mat.rows * mat.cols * mat.channels()); i += mat.channels())
        {
            for (int j = 0; j < mat.channels(); ++j)
                is_correct &= (((int *)(mat.data))[i+j] == (((int *)mat_other.data))[i+j]);
        }
    return is_correct;
    case CV_32F:
        for (int i = 0; (is_correct == true) && (i < mat.rows * mat.cols * mat.channels()); i += mat.channels())
        {
            for (int j = 0; j < mat.channels(); ++j)
                is_correct &= ((((float *)(mat.data))[i+j] - (((float *)mat_other.data))[i+j] >= -0.00001) || \
                                (((float *)(mat.data))[i+j] - (((float *)mat_other.data))[i+j] <= 0.00001));
        }
    return is_correct;
    case CV_64F:
        for (int i = 0; (is_correct == true) && (i < mat.rows * mat.cols * mat.channels()); i += mat.channels())
        {
            for (int j = 0; j < mat.channels(); ++j)
                is_correct &= ((((double *)(mat.data))[i+j] - (((double *)mat_other.data))[i+j] >= -0.00001) || \
                                (((double *)(mat.data))[i+j] - (((double *)mat_other.data))[i+j] <= 0.00001));
        }
    return is_correct;
    }
    return is_correct;
}

void Common_Test::MatShow(cv::Mat &m, string str)
{
    cout << str.c_str() << endl;
    cout << m;
    cout << endl
         << endl
         << endl;
}

void Common_Test::StatShow(cv::Mat &mat_src, aclMat &aclmat_dst)
{
    cout << "//////////////////////////////// MatStat ////////////////////////////////" << endl;
    cout << "type: " << mat_src.type() << endl;
    cout << "elemSize: " << mat_src.elemSize() << endl;
    cout << "channels: " << mat_src.channels() << endl;
    cout << "step: " << mat_src.step << endl;
    cout << "totalSize: " << mat_src.rows * mat_src.cols * mat_src.elemSize() << endl;
    cout << "totalSize: " << mat_src.total() * mat_src.elemSize() << endl;
    cout << "dataend - datastart: " << mat_src.dataend - mat_src.datastart << endl;

    cout << "//////////////////////////////// aclMatStat ////////////////////////////////" << endl;
    cout << "type: " << aclmat_dst.type() << endl;
    cout << "elemSize: " << aclmat_dst.elemSize() << endl;
    cout << "channels: " << aclmat_dst.channels() << endl;
    cout << "step: " << aclmat_dst.step << endl;
    cout << "totalSize: " << aclmat_dst.rows * aclmat_dst.step << endl;
    cout << "totalSize: " << aclmat_dst.totalSize << endl;
    cout << "dataend - datastart: " << aclmat_dst.dataend - aclmat_dst.datastart << endl;
    cout << "wholerows: " << aclmat_dst.wholerows << endl;
    cout << "wholecols: " << aclmat_dst.wholecols << endl;
    cout << "offset : " << aclmat_dst.offset << endl;
}

void Common_Test::PrintLog(const string& funcname, int type)
{
    switch (type)
    {
    case CV_8UC1:
        cout << funcname << "\t"
             << "Type: CV_8UC1" << endl;
        break;
    case CV_8UC3:
        cout << funcname << "\t"
             << "Type: CV_8UC3" << endl;
        break;
    case CV_32FC1:
        cout << funcname << "\t"
             << "Type: CV_32FC1" << endl;
        break;
    case CV_32FC3:
        cout << funcname << "\t"
             << "Type: CV_32FC3" << endl;
        break;
    case CV_32SC1:
        cout << funcname << "\t"
             << "Type: CV_32SC1" << endl;
        break;
    case CV_32SC3:
        cout << funcname << "\t"
             << "Type: CV_32SC3" << endl;
        break;
    case CV_64FC1:
        cout << funcname << "\t"
             << "Type: CV_64FC1" << endl;
        break;
    default:
        break;
    }
}

/* srand((unsigned)time(NULL)) in constructor */
size_t Common_Test::RandDom_(int config) {
    return static_cast<size_t>(rand() % config);
}

bool Common_Test::SetDataRange(Mat &src, int dataRange)
{
    switch (src.depth()) 
    {
    case CV_8U:
        for (int i = 0; i < src.rows * src.cols * src.channels(); i += src.channels())
        {
            for (int j = 0; j < src.channels(); ++j)
                (src.data)[i+j] = RandDom_(dataRange);
        }
        return true;
    case CV_16U:
        for (int i = 0; i < src.rows * src.cols * src.channels(); i += src.channels())
        {
            for (int j = 0; j < src.channels(); ++j)
                ((unsigned short *)src.data)[i+j] = RandDom_(dataRange);
        }
        return true;
    case CV_32S:
        for (int i = 0; i < src.rows * src.cols * src.channels(); i += src.channels())
        {
            for (int j = 0; j < src.channels(); ++j)
                ((int *)src.data)[i+j] = RandDom_(dataRange);
        }
        return true;
    case CV_32F:
        for (int i = 0; i < src.rows * src.cols * src.channels(); i += src.channels())
        {
            for (int j = 0; j < src.channels(); ++j)
                ((float *)src.data)[i+j] = RandDom_(dataRange) / 1.0;
        }
        return true;
    case CV_64F:
        for (int i = 0; i < src.rows * src.cols * src.channels(); i += src.channels())
        {
            for (int j = 0; j < src.channels(); ++j)
                ((double *)src.data)[i+j] = RandDom_(dataRange) / 1.0;
        }
        return true;
    default:
        return false;
    }
}
