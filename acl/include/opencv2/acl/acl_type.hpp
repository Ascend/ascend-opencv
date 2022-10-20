/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPENCV_ACL_TYPE_HPP
#define OPENCV_ACL_TYPE_HPP

#define AclSafeCall(expr) aclSafeCall(expr, __FILE__, __LINE__, __func__)
#define AclVerifyCall(expr) aclSafeCall(res, __FILE__, __LINE__, __func__)

#include <iostream>

#include "acl/acl.h"
#include "opencv2/core.hpp"

namespace cv {
namespace acl {
/**
 * An error is reported if the expression value is not 0
 */
inline void aclSafeCall(int err, const char *file, const int line,
                        const char *func = "") {
  if (0 != err) {
    const char *function = func ? func : "unknown function";
    std::cerr << "Acl Called Error: "
              << "file " << file << ", func " << function << ", line " << line
              << " errorCode: " << err << std::endl;
    std::cerr.flush();
  }
}

/* Memory alignment */
enum ALIGNMENT { MEMORY_UNALIGNED = 0, MEMORY_ALIGN = 1 };

enum {
  MAGIC_VAL = 0x42FF0000,
  AUTO_STEP = 0,
  CONTINUOUS_FLAG = CV_MAT_CONT_FLAG,
  SUBMATRIX_FLAG = CV_SUBMAT_FLAG
};
enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

using aclStream = aclrtStream;

using Opdims = enum Opdims { TWO_DIMS = 1, FOUR_DIMS };

enum DeviceType {
  ACL_DEVICE_TYPE_DEFAULT = (1 << 0),
  ACL_DEVICE_TYPE_200 = (1 << 1),
  ACL_DEVICE_TYPE_ACCELERATOR = (1 << 3),
};

enum AttrType { OP_BOOL = 1, OP_INT, OP_FLOAT, OP_STRING };

using MemMallocPolicy = enum MemMallocPolicy {
  MALLOC_HUGE_FIRST = 1,
  MALLOC_HUGE_ONLY,
  MALLOC_NORMAL_ONLY,
  MALLOC_HUGE_FIRST_P2P,
  MALLOC_HUGE_ONLY_P2P,
  MALLOC_NORMAL_ONLY_P2P
};

CV_EXPORTS aclDataType type_transition(int depth);
CV_EXPORTS aclrtMemMallocPolicy type_transition(MemMallocPolicy type);

inline aclDataType type_transition(int depth) {
  switch (depth) {
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
    default:
      return ACL_DT_UNDEFINED;
  }
}

inline aclrtMemMallocPolicy type_transition(MemMallocPolicy type) {
  switch (type) {
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
    default:
      return ACL_MEM_MALLOC_HUGE_FIRST;
  }
}
} /* end of namespace acl */
} /* end of namespace cv */

#endif /* __OPENCV_ACL_HPP__ */
