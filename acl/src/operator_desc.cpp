
/**
* @file operator_desc.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "precomp.hpp"
#include "../include/opencv2/acl/operator_desc.hpp"

using namespace std;

namespace cv
{
    namespace acl
    {
        OperatorDesc::OperatorDesc(std::string opType) : opType(std::move(opType))
        {
            opAttr = aclopCreateAttr();
        }

        OperatorDesc::~OperatorDesc()
        {
            for (auto *desc : inputDesc)
            {
                aclDestroyTensorDesc(desc);
            }

            for (auto *desc : outputDesc)
            {
                aclDestroyTensorDesc(desc);
            }

            aclopDestroyAttr(opAttr);
        }

        OperatorDesc &OperatorDesc::AddInputTensorDesc(aclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aclFormat format)
        {
            aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
            CV_Assert(desc);
            inputDesc.emplace_back(desc);
            return *this;
        }

        OperatorDesc &OperatorDesc::AddOutputTensorDesc(aclDataType dataType,
                                                        int numDims,
                                                        const int64_t *dims,
                                                        aclFormat format)
        {
            aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
            CV_Assert(desc);
            outputDesc.emplace_back(desc);
            return *this;
        }

        aclDataType type_transition(int type)
        {
            switch (type)
            {
            case CV_8UC1:
            case CV_8UC2:
            case CV_8UC3:
            case CV_8UC4:
                return ACL_UINT8;
            case CV_8SC1:
            case CV_8SC2:
            case CV_8SC3:
            case CV_8SC4:
                return ACL_INT8;
            case CV_16UC1:
            case CV_16UC2:
            case CV_16UC3:
            case CV_16UC4:
                return ACL_UINT16;
            case CV_16SC1:
            case CV_16SC2:
            case CV_16SC3:
            case CV_16SC4:
                return ACL_INT16;
            case CV_16FC1:
            case CV_16FC2:
            case CV_16FC3:
            case CV_16FC4:
                return ACL_FLOAT16;
            case CV_32SC1:
            case CV_32SC2:
            case CV_32SC3:
            case CV_32SC4:
                return ACL_INT32;
            case CV_32FC1:
            case CV_32FC2:
            case CV_32FC3:
            case CV_32FC4:
                return ACL_FLOAT;
            case CV_64FC1:
            case CV_64FC2:
            case CV_64FC3:
            case CV_64FC4:
                return ACL_DOUBLE;
            }
            return ACL_DT_UNDEFINED;
        }

    }
}
