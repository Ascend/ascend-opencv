#ifndef OPERATOR_DESC_HPP__
#define OPERATOR_DESC_HPP__

#include <string>
#include <vector>

#include "acl/acl.h"

namespace cv
{
       namespace acl
       {
              enum AttrType
              {
                     OP_BOOL = 1,
                     OP_INT,
                     OP_FLOAT,
                     OP_STRING
              };

              class CV_EXPORTS OperatorDesc
              {
              public:
                     /**
                      * Constructor
                      * @param [in] opType: op type
                      */
                     OperatorDesc(std::string opType);

                     /**
                      * Destructor
                      */
                     virtual ~OperatorDesc();

                     /**
                      * Add an input tensor description
                      * @param [in] dataType: data type
                      * @param [in] numDims: number of dims
                      * @param [in] dims: dims
                      * @param [in] format: format
                      * @return OperatorDesc
                      */
                     OperatorDesc &AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

                     /**
                      * Add an output tensor description
                      * @param [in] dataType: data type
                      * @param [in] numDims: number of dims
                      * @param [in] dims: dims
                      * @param [in] format: format
                      * @return OperatorDesc
                      */
                     OperatorDesc &AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

                     template <typename T>
                     bool AddTensorAttr(const char *attrName, AttrType type, T vaule);
                     
                     std::string opType;
                     std::vector<aclTensorDesc *> inputDesc;
                     std::vector<aclTensorDesc *> outputDesc;
                     aclopAttr *opAttr;
              };

              CV_EXPORTS aclDataType type_transition(int type);

              template <typename T>
              bool OperatorDesc::AddTensorAttr(const char *attrName, AttrType type, T vaule)
              {
                     if (opAttr == nullptr)
                     {
                            return false;
                     }
                     switch (type)
                     {
                     case OP_BOOL:
                            aclopSetAttrBool(opAttr, attrName, vaule);
                            break;
                     case OP_INT:
                            aclopSetAttrInt(opAttr, attrName, vaule);
                            break;
                     case OP_FLOAT:
                            aclopSetAttrFloat(opAttr, attrName, vaule);
                            break;
                     default:
                            break;
                     }

                     return true;
              }
       }

}

#endif // OPERATOR_DESC_HPP
