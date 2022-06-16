#include "test_common.hpp"

/* thread function */
void thread_handler(void) {
    aclCxt *acl_context_0 = set_device("/home/perfxlab4/OpenCV_ACL/modules/acl/test/acl.json", 0, 1);
    release_device(acl_context_0);
}

AclMat_Test::AclMat_Test() {
    srand((unsigned)time(NULL));
}

AclMat_Test::~AclMat_Test() {

}

void AclMat_Test::Test_set_device() {
    /* Current thread */
    aclCxt *acl_context_0 = set_device("/home/perfxlab4/OpenCV_ACL/modules/acl/test/acl.json", 0, 1);

    /* Different scope */
    {
        aclCxt *acl_context_1 = set_device("/home/perfxlab4/OpenCV_ACL/modules/acl/test/acl.json", 2, 3);
        release_device(acl_context_1);
    }

    release_device(acl_context_0);
    /* Different thread */
    thread t(thread_handler);
    t.join();
}

void AclMat_Test::Test_constructor_UNALIGNED(aclCxt *acl_context) {

    int rows, cols, type;
    bool ret;
    const int rowsMax = 128;
    const int colsMax = 128;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                Mat mat_src(rows, cols, type);
                aclMat aclmat_src(rows, cols, type, acl_context);
                aclmat_src.upload(mat_src);
                ret = Test_Diff(aclmat_src, mat_src);     
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_constructor_UNALIGNED: -> aclMat(rows, cols, type, acl_context, config, policy) <- is success" << endl;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                Mat mat_src(cv::Size(cols, rows), type);
                aclMat aclmat_src(cv::Size(cols, rows), type, acl_context);
                aclmat_src.upload(mat_src);
                ret = Test_Diff(aclmat_src, mat_src);     
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_constructor_UNALIGNED: -> aclMat(size, type, acl_context, config, policy) <- is success" << endl;
}

void AclMat_Test::Test_constructor_ALIGN(aclCxt *acl_context) {

    int rows, cols, type;
    bool ret;
    const int rowsMax = 128;
    const int colsMax = 128;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                Mat mat_src(rows, cols, type);
                aclMat aclmat_src(rows, cols, type, acl_context, MEMORY_ALIGN);
                aclmat_src.upload(mat_src, MEMORY_ALIGN);
                ret = Test_Diff(aclmat_src, mat_src, MEMORY_ALIGN);     
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_constructor_ALIGN: -> aclMat(rows, cols, type, acl_context, config, policy) <- is success" << endl;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                Mat mat_src(cv::Size(cols, rows), type);
                aclMat aclmat_src(cv::Size(cols, rows), type, acl_context, MEMORY_ALIGN);
                aclmat_src.upload(mat_src, MEMORY_ALIGN);
                ret = Test_Diff(aclmat_src, mat_src, MEMORY_ALIGN);     
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_constructor_ALIGN: -> aclMat(size, type, acl_context, config, policy) <- is success" << endl;

    
}

void AclMat_Test::Test_constructor(aclCxt *acl_context_0) {
    int rows, cols, type;
    bool ret;
    const int rowsMax = 128;
    const int colsMax = 128;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                aclMat aclmat_src(rows, cols, type, acl_context_0);
                aclMat aclmat_dest(aclmat_src);
                ret = Test_Diff(aclmat_src, aclmat_dest); 
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_constructor: -> aclMat(aclmat_src) <- is success" << endl;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                aclMat aclmat_src(cv::Size(cols, rows), type, acl_context_0, MEMORY_ALIGN);
                aclMat aclmat_dest(aclmat_src);
                ret = Test_Diff(aclmat_src, aclmat_dest);     
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_constructor: -> aclMat(const aclMat& other) <- is success" << endl;
}

void AclMat_Test::Test_constructor_DATA(aclCxt *acl_context_0) {
    int rows, cols, type;
    bool ret;
    const int rowsMax = 128;
    const int colsMax = 128;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                Mat mat_src(rows, cols, type);
                Mat mat_dest(rows, cols, type);
                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
                aclmat_src.download(mat_dest);
                ret = Test_Diff(mat_src, mat_dest);     
                ASSERT_TRUE(ret);
            }
        }
    }
    cerr << "Test_constructor_DATA: -> aclMat(rows, cols, type, data, acl_context)) <- is success" << endl;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                Mat mat_src(cv::Size(cols, rows), type);
                Mat mat_dest(cv::Size(cols, rows), type);
                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                aclMat aclmat_src(cv::Size(cols, rows), type, mat_src.data, acl_context_0);
                aclmat_src.download(mat_dest);
                ret = Test_Diff(mat_src, mat_dest);     
                ASSERT_TRUE(ret);
            }
        }
    }

    cerr << "Test_constructor_DATA: -> aclMat(size, type, data, acl_context)) <- is success" << endl;
}

void AclMat_Test::Test_constructor_RANGE(aclCxt *acl_context_0) {
    int type;
    bool ret;
    int rangerows, rangecols;
    int rows = 64, cols = 64;
    const int rangerowsMax = 64;
    const int rangecolsMax = 64;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++) {
        for (rangerows = 4; rangerows < rangerowsMax; rangerows++) {
            for (rangecols = 4; rangecols < rangecolsMax; rangecols++) {
                Mat mat_src(rows, cols, type);
                Mat mat_dest(rows, cols, type);
                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
                    mat_dest.data[i] = RandDom_();
                Mat mat_rangesrc(mat_src, cv::Range(2, rangerows), cv::Range(2, rangecols));
                Mat mat_rangedest(mat_dest, cv::Range(2, rangerows), cv::Range(2, rangecols));
                aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
                aclMat aclmat_range(aclmat_src, cv::Range(2, rangerows), cv::Range(2, rangecols));
                aclmat_range.download(mat_rangedest);
                ret = Test_Diff(mat_rangesrc, mat_rangedest);     
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_constructor_RANGE: -> aclMat(aclmat_src, rowragne, colrange)) <- is success" << endl;

}

void AclMat_Test::Test_constructor_ROI(aclCxt *acl_context_0) {
    {
        int rows = 6, cols = 8;
        int type = CV_8UC1;
        cv::Rect roi(2, 2, 1, 1);
        bool ret;
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);

        for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
            mat_src.data[i] = RandDom_();
        for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
            mat_dest.data[i] = RandDom_();

        Mat mat_roi1(mat_src, roi);
        Mat mat_roi(mat_dest, roi);

        aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
        aclMat aclmat_roi(aclmat_src, roi);
        aclmat_roi.download(mat_roi);
        ret = Test_Diff(mat_roi1, mat_roi);
        ASSERT_TRUE(ret);
    }

    {
        int rows = 12, cols = 61;
        int type = CV_16UC3;
        cv::Rect roi(8, 8, 2, 2);
        bool ret;
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);

        for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
            mat_src.data[i] = RandDom_();
        for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
            mat_dest.data[i] = RandDom_();

        Mat mat_roi1(mat_src, roi);
        Mat mat_roi(mat_dest, roi);

        aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
        aclMat aclmat_roi(aclmat_src, roi);
        aclmat_roi.download(mat_roi);
        ret = Test_Diff(mat_roi1, mat_roi);
        ASSERT_TRUE(ret);
    }

    {
        int rows = 16, cols = 80;
        int type = CV_32FC3;
        cv::Rect roi(8, 4, 1, 3);
        bool ret;
        Mat mat_src(rows, cols, type);
        Mat mat_dest(rows, cols, type);

        for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
            mat_src.data[i] = RandDom_();
        for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
            mat_dest.data[i] = RandDom_();

        Mat mat_roi1(mat_src, roi);
        Mat mat_roi(mat_dest, roi);

        aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
        aclMat aclmat_roi(aclmat_src, roi);
        aclmat_roi.download(mat_roi);
        ret = Test_Diff(mat_roi1, mat_roi);
        ASSERT_TRUE(ret);
    }

    clog << "Test_constructor_ROI: -> aclMat(aclmat_src, roi)) <- is success" << endl;
}

void AclMat_Test::Test_constructor_MAT(aclCxt *acl_context_0) {
    int rows, cols, type;
    bool ret;
    const int rowsMax = 128;
    const int colsMax = 128;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++) {
        for (rows = 1; rows < rowsMax; rows++) {
            for (cols = 1; cols < colsMax; cols++) {
                Mat mat_src(rows, cols, type);
                Mat mat_dest(rows, cols, type);
                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                aclMat aclmat_src(mat_src, acl_context_0);
                aclmat_src.download(mat_dest);
                ret = Test_Diff(mat_src, mat_dest);
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_constructor_MAT: -> aclMat(mat_src, acl_context_0)) <- is success" << endl;

}

void AclMat_Test::Test_DATA_TRANSFER(aclCxt *acl_context_0) {
    int rows, cols, type;
    bool ret;
    const int rowsMax = 128;
    const int colsMax = 128;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++)
    {
        for (rows = 1; rows < rowsMax; rows++)
        {
            for (cols = 1; cols < colsMax; cols++)
            {
                Mat mat_src(rows, cols, type);
                Mat mat_dest(rows, cols, type);
                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
                    mat_dest.data[i] = RandDom_();
                aclMat aclmat_src(rows, cols, type, acl_context_0);
                aclmat_src.upload(mat_src);
                aclmat_src.download(mat_dest);
                ret = Test_Diff(mat_src, mat_dest);
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_DATA_TRANSFER_UNALIGNED: -> upload(), download() <- is success" << endl;

    for (type = 0; type < typeMax; type++)
    {
        for (rows = 1; rows < rowsMax; rows++)
        {
            for (cols = 1; cols < colsMax; cols++)
            {
                Mat mat_src(rows, cols, type);
                Mat mat_dest(rows, cols, type);
                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
                    mat_dest.data[i] = RandDom_();
                aclMat aclmat_src(rows, cols, type, acl_context_0, MEMORY_ALIGN);
                aclmat_src.upload(mat_src, MEMORY_ALIGN);
                aclmat_src.download(mat_dest, MEMORY_ALIGN);
                ret = Test_Diff(mat_src, mat_dest);
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_DATA_TRANSFER_ALIGN: -> upload(), download() <- is success" << endl;
}

void AclMat_Test::Test_DATA_TRANSFERASYNC(aclCxt *acl_context_0) {
    int rows, cols, type;
    bool ret;
    const int rowsMax = 128;
    const int colsMax = 128;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++)
    {
        for (rows = 1; rows < rowsMax; rows++)
        {
            for (cols = 1; cols < colsMax; cols++)
            {
                Mat mat_src(rows, cols, type);
                Mat mat_dest(rows, cols, type);
                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
                    mat_dest.data[i] = RandDom_();
                aclMat aclmat_src(rows, cols, type, acl_context_0);
                aclmat_src.upload(mat_src, aclmat_src.acl_context->get_stream(0));
                aclmat_src.download(mat_dest, aclmat_src.acl_context->get_stream(0));
                ret = Test_Diff(mat_src, mat_dest);
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_DATA_TRANSFERASYNC_UNALIGNED: -> upload(), download() <- is success" << endl;

    for (type = 0; type < typeMax; type++)
    {
        for (rows = 1; rows < rowsMax; rows++)
        {
            for (cols = 1; cols < colsMax; cols++)
            {
                Mat mat_src(rows, cols, type);
                Mat mat_dest(rows, cols, type);
                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
                    mat_dest.data[i] = RandDom_();
                aclMat aclmat_src(rows, cols, type, acl_context_0, MEMORY_ALIGN);
                aclmat_src.upload(mat_src, aclmat_src.acl_context->get_stream(1), MEMORY_ALIGN);
                aclmat_src.download(mat_dest, aclmat_src.acl_context->get_stream(1), MEMORY_ALIGN);
                ret = Test_Diff(mat_src, mat_dest);
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_DATA_TRANSFERASYNC_ALIGN: -> upload(), download() <- is success" << endl;
   
}

static inline void dataSwap(int& data1, int& data2) {
    int temp;
    if (data1 < data2) {
        temp = data1;
        data1 = data2;
        data2 = temp;
    }
}

void AclMat_Test::Test_locateROI(aclCxt *acl_context_0) {
    int rows = 256, cols = 256;
    int type = CV_8UC1;
    int rangex, rangey;
    int rangex1, rangey1;
    cv::Size size, size1;
    cv::Point ofs, ofs1;

    for (int x = 0; x < rows * cols; ++x)
    {

        rangex = (rangex = RandDom_()) > 0 ? rangex : 1;
        rangey = (rangey = RandDom_()) > 0 ? rangey : 1;
        rangex1 = (rangex1 = RandDom_()) > 0 ? rangex1 : 1;
        rangey1 = (rangey1 = RandDom_()) > 0 ? rangey1 : 1;

        dataSwap(rangex, rangex1);
        dataSwap(rangey, rangey1);

        Mat mat_src(rows, cols, type);
        Mat mat_range(mat_src, cv::Range(rangex1, rangex+1), cv::Range(rangey1, rangey+1));
        mat_range.locateROI(size, ofs);

        aclMat aclmat_src(rows, cols, type, acl_context_0);
        aclMat aclmat_range(aclmat_src, cv::Range(rangex1, rangex+1), cv::Range(rangey1, rangey+1));
        aclmat_range.locateROI(size1, ofs1);

        ASSERT_EQ(size.height, size1.height);
        ASSERT_EQ(size.width, size1.width);
        ASSERT_EQ(ofs.x, ofs1.x);
        ASSERT_EQ(ofs.y, ofs1.y);
    }
    clog << "Test_loacteROI: -> locateROI() <- is success" << endl;
    
}

void AclMat_Test::Test_swap(aclCxt *acl_context_0) {
    int rows, cols, type;
    bool ret;
    const int rowsMax = 128;
    const int colsMax = 128;
    const int typeMax = 7;

    for (type = 0; type < typeMax; type++)
    {
        for (rows = 1; rows < rowsMax; rows++)
        {
            for (cols = 1; cols < colsMax; cols++)
            {
                Mat mat_src(rows, cols, type);
                Mat mat_dest(rows, cols, type);

                for (size_t i = 0; i < mat_src.rows * mat_src.cols * mat_src.elemSize(); i++)
                    mat_src.data[i] = RandDom_();
                for (size_t i = 0; i < mat_dest.rows * mat_dest.cols * mat_dest.elemSize(); i++)
                    mat_dest.data[i] = RandDom_();
                Mat mat_dest1(rows, cols, type);
                Mat mat_dest2(rows, cols, type);

                aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context_0);
                aclMat aclmat_src1(rows, cols, type, mat_dest.data, acl_context_0);
                aclmat_src.swap(aclmat_src1);

                aclmat_src.download(mat_dest1);
                aclmat_src1.download(mat_dest2);

                ret = Test_Diff(mat_dest1, mat_dest);
                ASSERT_TRUE(ret);

                ret = Test_Diff(mat_dest2, mat_src);
                ASSERT_TRUE(ret);
            }
        }
    }
    clog << "Test_Swap: -> swap() <- is success" << endl;
}

void AclMat_Test::Test_operator(aclCxt *acl_context) {
    int rows, cols, type;
    bool ret;
    const int rowsMax = 8;
    const int colsMax = 8;

    type = CV_8UC1;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                mat_src.data[i] = RandDom_(100);
                mat_dest.data[i] = RandDom_(100);
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_8UC1) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                mat_src.data[i] = RandDom_(15);
                mat_dest.data[i] = RandDom_(100) + 20;
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);
            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_8UC1) <- is success" << endl;

    type = CV_8UC3;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i += 3)
            {
                mat_src.data[i] = RandDom_(100);
                mat_src.data[i+1] = RandDom_(100);
                mat_src.data[i+2] = RandDom_(100);
                mat_dest.data[i] = RandDom_(100);
                mat_dest.data[i+1] = RandDom_(100);
                mat_dest.data[i+2] = RandDom_(100);
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_8UC3) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                mat_src.data[i] = RandDom_(15);
                mat_src.data[i+1] = RandDom_(15);
                mat_src.data[i+2] = RandDom_(15);
                mat_dest.data[i] = RandDom_(100) + 20;
                mat_dest.data[i+1] = RandDom_(100) + 20;
                mat_dest.data[i+2] = RandDom_(100) + 20;
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);
            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_8UC3) <- is success" << endl;

    type = CV_16UC1;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                ((unsigned short *)mat_src.data)[i] = RandDom_(100);
                ((unsigned short *)mat_dest.data)[i] = RandDom_(100);
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_16UC1) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                ((unsigned short *)mat_src.data)[i] = RandDom_(15);
                ((unsigned short *)mat_dest.data)[i] = RandDom_(100) + 20;
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);
            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_16UC1) <- is success" << endl;

    type = CV_16UC3;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i += 3)
            {
                ((unsigned short *)mat_src.data)[i] = RandDom_(100);
                ((unsigned short *)mat_src.data)[i+1] = RandDom_(100);
                ((unsigned short *)mat_src.data)[i+2] = RandDom_(100);
                ((unsigned short *)mat_dest.data)[i] = RandDom_(100);
                ((unsigned short *)mat_dest.data)[i+1] = RandDom_(100);
                ((unsigned short *)mat_dest.data)[i+2] = RandDom_(100);
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_16UC3) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                ((unsigned short *)mat_src.data)[i] = RandDom_(15);
                ((unsigned short *)mat_src.data)[i+1] = RandDom_(15);
                ((unsigned short *)mat_src.data)[i+2] = RandDom_(15);
                ((unsigned short *)mat_dest.data)[i] = RandDom_(100) + 20;
                ((unsigned short *)mat_dest.data)[i+1] = RandDom_(100) + 20;
                ((unsigned short *)mat_dest.data)[i+2] = RandDom_(100) + 20;
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);
            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_16UC3) <- is success" << endl;

    type = CV_32SC1;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                ((int *)mat_src.data)[i] = RandDom_(100);
                ((int *)mat_dest.data)[i] = RandDom_(100);
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_32SC1) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                ((int *)mat_src.data)[i] = RandDom_(15);
                ((int *)mat_dest.data)[i] = RandDom_(100) + 20;
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);
            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_32SC1) <- is success" << endl;

    type = CV_32SC3;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i += 3)
            {
                ((int *)mat_src.data)[i] = RandDom_(100);
                ((int *)mat_src.data)[i+1] = RandDom_(100);
                ((int *)mat_src.data)[i+2] = RandDom_(100);
                ((int *)mat_dest.data)[i] = RandDom_(100);
                ((int *)mat_dest.data)[i+1] = RandDom_(100);
                ((int *)mat_dest.data)[i+2] = RandDom_(100);
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_32SC3) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type);
            Mat mat_dest(rows, cols, type);
            Mat mat_dest1(rows, cols, type);

            for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
            {
                ((int *)mat_src.data)[i] = RandDom_(15);
                ((int *)mat_src.data)[i+1] = RandDom_(15);
                ((int *)mat_src.data)[i+2] = RandDom_(15);
                ((int *)mat_dest.data)[i] = RandDom_(100) + 20;
                ((int *)mat_dest.data)[i+1] = RandDom_(100) + 20;
                ((int *)mat_dest.data)[i+2] = RandDom_(100) + 20;
            }

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);
            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_32SC3) <- is success" << endl;

    type = CV_32FC1;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type, Scalar(2.0));
            Mat mat_dest(rows, cols, type, Scalar(4.0));
            Mat mat_dest1(rows, cols, type, Scalar(8.0));

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);
            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_32FC1) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type, Scalar(2.0));
            Mat mat_dest(rows, cols, type, Scalar(4.0));
            Mat mat_dest1(rows, cols, type, Scalar(8.0));

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_32FC1) <- is success" << endl;

    type = CV_32FC3;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type, Scalar(2.0, 3.0, 4.0));
            Mat mat_dest(rows, cols, type, Scalar(4.0, 5.0, 6.0));
            Mat mat_dest1(rows, cols, type, Scalar(8.0, 9.0, 10.0));

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_32FC3) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type, Scalar(2.0, 3.0, 4.0));
            Mat mat_dest(rows, cols, type, Scalar(4.0, 5.0, 6.0));
            Mat mat_dest1(rows, cols, type, Scalar(8.0, 9.0, 10.0));
            
            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_32FC3) <- is success" << endl;

    type = CV_64FC1;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type, Scalar(2.0));
            Mat mat_dest(rows, cols, type, Scalar(4.0));
            Mat mat_dest1(rows, cols, type, Scalar(8.0));

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);
            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_64FC1) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type, Scalar(2.0));
            Mat mat_dest(rows, cols, type, Scalar(4.0));
            Mat mat_dest1(rows, cols, type, Scalar(8.0));

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_64FC1) <- is success" << endl;

    type = CV_64FC3;
    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type, Scalar(2.0, 3.0, 4.0));
            Mat mat_dest(rows, cols, type, Scalar(4.0, 5.0, 6.0));
            Mat mat_dest1(rows, cols, type, Scalar(8.0, 9.0, 10.0));

            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest += mat_src;
            aclmat_dest += aclmat_src;
            aclmat_dest.download(mat_dest1);

            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator+=(type = CV_64FC3) <- is success" << endl;

    for (rows = 1; rows < rowsMax; rows++)
    {
        for (cols = 1; cols < colsMax; cols++)
        {
            Mat mat_src(rows, cols, type, Scalar(2.0, 3.0, 4.0));
            Mat mat_dest(rows, cols, type, Scalar(4.0, 5.0, 6.0));
            Mat mat_dest1(rows, cols, type, Scalar(8.0, 9.0, 10.0));
            
            aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
            aclMat aclmat_dest(rows, cols, type, mat_dest.data, acl_context);

            mat_dest -= mat_src;
            aclmat_dest -= aclmat_src;
            aclmat_dest.download(mat_dest1);
            ret = Test_Diff(mat_dest, mat_dest1);
            ASSERT_TRUE(ret);
        }
    }
    clog << "Test_operator: -> operator-=(type = CV_64FC3) <- is success" << endl;


}


void AclMat_Test::Test_operator_perf(aclCxt *acl_context) {
    int val, type;
    int valmax = 8192;
    double begin, end, time, acltime;

    type = CV_32FC1;

    for (val = 8; val <= valmax; val *= 2)
    {
        int n = 100;
        Mat mat_src(val, val, type, Scalar{4});
        Mat mat_dest(val, val, type, Scalar{6});
        Mat mat_dest1(val, val, type, Scalar{2});
        aclMat aclmat_src(val, val, type, mat_src.data, acl_context);
        aclMat aclmat_dest(val, val, type, mat_dest.data, acl_context);


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
        bool ret = Test_Diff(mat_dest, mat_dest1);
        ASSERT_TRUE(ret);

        if (val < 128)
            cout << "Shape: " << val << " x " << val << "\t\t";
        else
            cout << "Shape: " << val << " x " << val << "\t";
        cout << "CpuTimes: " << time << "\tAclTimes: " << acltime << "\tRate: " << time / acltime << endl;
    }
}

void AclMat_Test::Test_Abs(aclCxt *acl_context) {
    int rows, cols, type;
    bool ret;
    const int rowsMax = 8;
    const int colsMax = 8;

    type = CV_32SC1;
    rows = 6, cols = 8;
    Mat mat_src(rows, cols, type);
    Mat mat_dest(rows, cols, type);
    for (int i = 0; i < mat_src.rows * mat_src.cols * mat_src.channels(); i++)
    {
        ((int *)mat_src.data)[i] = RandDom_(100) - 100;
        ((int *)mat_dest.data)[i] = RandDom_(100) - 100;
    }

    aclMat aclmat_src(rows, cols, type, mat_src.data, acl_context);
    cout << mat_src << endl;
    aclmat_src.abs();
    aclmat_src.download(mat_dest);

    cout << mat_dest << endl;
}

bool AclMat_Test::Test_Diff(const aclMat& aclmat, const Mat& mat, ALIGNMENT config) {
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
    
bool AclMat_Test::Test_Diff(const aclMat& aclmat, const aclMat& aclmat_other) {
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

bool AclMat_Test::Test_Diff(const Mat &mat, const Mat &mat_other)
{
    bool is_correct;
    
    is_correct = (mat.rows == mat_other.rows);
    is_correct &= (mat.cols == mat_other.cols);
    is_correct &= (mat.type() == mat_other.type());
    is_correct &= (mat.channels() == mat.channels());
    is_correct &= (mat.step == mat_other.step);
    is_correct &= (mat.elemSize() == mat_other.elemSize());
    is_correct &= (mat.total() == mat_other.total());

    if (is_correct) {
        for (size_t i = 0; i < mat.rows * mat.cols * mat.elemSize(); i++) {
            is_correct &= (mat.data[i] == mat_other.data[i]);
        }
    }
    return is_correct;
}

void AclMat_Test::MatShow(cv::Mat &m, string str)
{
    cout << str.c_str() << endl;
    cout << m;
    cout << endl
         << endl
         << endl;
}

void AclMat_Test::StatShow(cv::Mat &mat_src, aclMat &aclmat_dst)
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

size_t AclMat_Test::RandDom_(int config)
{
    /* srand((unsigned)time(NULL)) in constructor */
    return static_cast<size_t>(rand() % config);
}