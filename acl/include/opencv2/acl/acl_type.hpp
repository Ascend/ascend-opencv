/*M///////////////////////////////////////////////////////////////////////////////////////
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

#ifndef OPENCV_ACL_TYPE_HPP
#define OPENCV_ACL_TYPE_HPP

#define AclSafeCall(expr) __aclSafeCall(expr, __FILE__, __LINE__, __func__)
#define AclVerifyCall(expr) __aclSafeCall(res, __FILE__, __LINE__, __func__)

#include <iostream>
#include "opencv2/core.hpp"
#include "acl/acl.h"

namespace cv
{
    namespace acl 
    {
        /**
         * An error is reported if the expression value is not 0
         */
        static inline void __aclSafeCall(int err, const char* file, const int line, const char *func="")
        {
            if(0 !=  err)
            {
                const char* function = func ? func : "unknown function";
                std::cerr << "Acl Called Error: " << "file " << file  << ", func " << function << ", line " << line << " errorCode: " << err << std::endl;
                std::cerr.flush();
            }
        }

        /* Memory alignment */
        enum ALIGNMENT { MEMORY_UNALIGNED = 0, MEMORY_ALIGN = 1};

        enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = CV_MAT_CONT_FLAG, SUBMATRIX_FLAG = CV_SUBMAT_FLAG };
        enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

        typedef aclrtStream aclStream;

        typedef enum Opdims { TWO_DIMS = 1, FOUR_DIMS } Opdims;

        enum DeviceType
        {
            ACL_DEVICE_TYPE_DEFAULT = (1 << 0),
            ACL_DEVICE_TYPE_200 = (1 << 1),
            ACL_DEVICE_TYPE_ACCELERATOR = (1 << 3),
        };

        enum AttrType
        {
            OP_BOOL = 1,
            OP_INT,
            OP_FLOAT,
            OP_STRING
        };

        typedef enum MemMallocPolicy
        {
            MALLOC_HUGE_FIRST = 1, 
            MALLOC_HUGE_ONLY,
            MALLOC_NORMAL_ONLY, 
            MALLOC_HUGE_FIRST_P2P, 
            MALLOC_HUGE_ONLY_P2P,  
            MALLOC_NORMAL_ONLY_P2P
        } MemMallocPolicy;


        CV_EXPORTS aclDataType type_transition(int depth);
        CV_EXPORTS aclrtMemMallocPolicy type_transition(MemMallocPolicy type);

        inline aclDataType type_transition(int depth)
        {
            switch (depth)
            {
            case CV_8U:
                return ACL_UINT8;
            case CV_8S:
                return ACL_INT8;
            case CV_16U:
                return ACL_UINT16;
            case CV_16S:
                return ACL_INT16;
            case CV_16F:
                return ACL_FLOAT16;
            case CV_32S:
                return ACL_INT32;
            case CV_32F:
                return ACL_FLOAT;
            case CV_64F:
                return ACL_DOUBLE;
            }
            return ACL_DT_UNDEFINED;
        }

        inline aclrtMemMallocPolicy type_transition(MemMallocPolicy type)
        {
            switch (type)
            {
            case MALLOC_HUGE_FIRST:
                return ACL_MEM_MALLOC_HUGE_FIRST;
            case MALLOC_HUGE_ONLY:
                return ACL_MEM_MALLOC_HUGE_ONLY;
            case MALLOC_NORMAL_ONLY:
                return ACL_MEM_MALLOC_NORMAL_ONLY;
            case MALLOC_HUGE_FIRST_P2P:
                return ACL_MEM_MALLOC_HUGE_FIRST_P2P;
            case MALLOC_HUGE_ONLY_P2P:
                return ACL_MEM_MALLOC_HUGE_ONLY_P2P;
            case MALLOC_NORMAL_ONLY_P2P:
                return ACL_MEM_MALLOC_NORMAL_ONLY_P2P;
            }
            return ACL_MEM_MALLOC_HUGE_FIRST;
        }

    } /* end of namespace acl */

} /* end of namespace cv */

#endif /* __OPENCV_ACL_HPP__ */
