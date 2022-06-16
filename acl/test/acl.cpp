
///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/



#include "test_precomp.hpp"
#include "test_common.hpp"


namespace opencv_test 
{
    namespace 
    {
        aclCxt *acl_context_0 = set_device("/home/perfxlab4/OpenCV_ACL/modules/acl/test/acl.json", 3);
        /* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7 
         * test function:
         * config: MEMORY_ALIGN
         * aclMat(int rows, int cols, int type, aclCxt *acl_context, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
         * aclMat(Size size, int type, aclCxt *acl_context, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
         * aclMat(const aclMat &m);
         * 
         */
#if 0
        TEST(ACLMAT_CONSTRUCTOR, MEMORY_ALIGN)
        {
            AclMat_Test test;
            test.Test_constructor_ALIGN(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7 
         * test function:
         * config: MEMORY_UNALIGNED
         * aclMat(int rows, int cols, int type, aclCxt *acl_context, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
         * aclMat(Size size, int type, aclCxt *acl_context, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
         * 
         */
        TEST(ACLMAT_CONSTRUCTOR, MEMORY_UNALIGNED)
        {
            AclMat_Test test;
            test.Test_constructor_UNALIGNED(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7 
         * test function:
         * aclMat(const aclMat &m);
         */
        TEST(ACLMAT_CONSTRUCTOR, COPY_CONSTRUCTOR)
        {
            AclMat_Test test;
            test.Test_constructor(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7 
         * test function:
         * aclMat(int rows, int cols, int type, void *data, aclCxt* acl_context, ALIGNMENT config = MEMORY_UNALIGNED, size_t step = Mat::AUTO_STEP);
         * aclMat(Size size, int type, void *data, aclCxt* acl_context, ALIGNMENT config = MEMORY_UNALIGNED, size_t step = Mat::AUTO_STEP);
         */
        TEST(ACLMAT_CONSTRUCTOR, DATA)
        {
            AclMat_Test test;
            test.Test_constructor_DATA(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7 
         * test function:
         * aclMat(const aclMat &m, const Range &rowRange, const Range &colRange = Range::all());
         * 
         */
        TEST(ACLMAT_CONSTRUCTOR, RANGE)
        {
            AclMat_Test test;
            test.Test_constructor_RANGE(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /*
         * test function:
         * aclMat(const aclMat &m, const Rect &roi);
         * 
         */
        TEST(ACLMAT_CONSTRUCTOR, ROI)
        {
            AclMat_Test test;
            test.Test_constructor_ROI(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }
        
        /*
         * test function: 
         * aclMat (const Mat &m, aclCxt* acl_context, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
         */
        TEST(ACLMAT_CONSTRUCTOR, MAT)
        {
            AclMat_Test test;
            test.Test_constructor_MAT(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7 
         * test function:
         * CV_EXPORTS void upload(const Mat &m, ALIGNMENT config = MEMORY_UNALIGNED);
         * CV_EXPORTS void upload(const Mat &m, aclStream stream, ALIGNMENT config = MEMORY_UNALIGNED);
         * 
         */
        TEST(ACLMAT_FUNCTION, DATA_TRANSFER)
        {
            AclMat_Test test;
            test.Test_DATA_TRANSFER(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /* range: rows: 1 ~ 64, cols: 1 ~ 64, type: 0 ~ 7 
         * test function:
         * CV_EXPORTS void download(Mat &m, ALIGNMENT config = MEMORY_UNALIGNED) const;
         * CV_EXPORTS void download(Mat &m, aclStream stream, ALIGNMENT config = MEMORY_UNALIGNED) const;
         * 
         */
        TEST(ACLMAT_FUNCTION, DATA_TRANSFERASYNC)
        {
            AclMat_Test test;
            test.Test_DATA_TRANSFERASYNC(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /*
         * test function:
         * void locateROI(Size &wholeSize, Point &ofs) const;
         */
        TEST(ACLMAT_FUNCTION, LOCATEROI)
        {
            AclMat_Test test;
            test.Test_locateROI(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        /*
         * test function:
         * void swap(aclMat &mat);
         * 
         */
        TEST(ACLMAT_FUNCTION, SWAP)
        {
            AclMat_Test test;
            test.Test_swap(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }
        /*
         * test function:
         * operator+=(), operator-=();
         * 
        */
        TEST(ACLMAT_FUNCTION, OPERATOR)
        {
            AclMat_Test test;
            test.Test_operator(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;
        }

        TEST(ACLMAT_FUNCTION, OPERATOR_PERF)
        {
            AclMat_Test test;
            test.Test_operator_perf(acl_context_0);
            cout << "///////////////////////////////////////////////////////////////////////////////" << endl;

        }
#endif
        TEST(ACLMAT_FUNCTION, Abs)
        {
            AclMat_Test test;
            test.Test_Abs(acl_context_0);

            release_device(acl_context_0);
        }
    }
}