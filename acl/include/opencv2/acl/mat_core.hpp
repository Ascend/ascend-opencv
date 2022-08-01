#ifndef OPENCV_MAT_CORE_HPP
#define OPENCV_MAT_CORE_HPP

#include "acl_type.hpp"

namespace cv
{
    namespace acl
    {
        ////////////////////////////////////////////////////////////////////////
        //////////////////////////////// aclMat ////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        inline aclMat::aclMat()
            : flags(0), rows(0), cols(0), step(0), data(nullptr), refcount(nullptr),
              datastart(nullptr), dataend(nullptr), offset(0), wholerows(0), wholecols(0), acl_context(0), totalSize(0)
        {
        }

        /**
         *   @param [in] _acl_context: Acl context
         *   @param [in] config: Byte aligned or not, Default MEMORY_UNALIGNED
         *   @param [in] policy: Acl Memory Application mode, Default ACL_MEM_MALLOC_NORMAL_ONLY
         */
        inline aclMat::aclMat(int _rows, int _cols, int _type, aclCxt *_acl_context, ALIGNMENT config, MemMallocPolicy policy)
            : flags(0), rows(0), cols(0), step(0), data(nullptr), refcount(nullptr), datastart(nullptr),
              dataend(nullptr), offset(0), wholerows(0), wholecols(0), acl_context(_acl_context), totalSize(0)
        {
            if (_rows > 0 && _cols > 0)
                create(_rows, _cols, _type, config, policy);
        }

        inline aclMat::aclMat(Size _size, int _type, aclCxt *_acl_context, ALIGNMENT config, MemMallocPolicy policy)
            : flags(0), rows(0), cols(0), step(0), data(nullptr), refcount(nullptr), datastart(nullptr),
              dataend(nullptr), offset(0), wholerows(0), wholecols(0), acl_context(_acl_context), totalSize(0)
        {
            if (_size.height > 0 && _size.width > 0)
                create(_size, _type, config, policy);
        }

        inline aclMat::aclMat(const aclMat &m)
            : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount),
              datastart(m.datastart), dataend(m.dataend), offset(m.offset), wholerows(m.wholerows), wholecols(m.wholecols),
              acl_context(m.acl_context), totalSize(m.totalSize)
        {
            if (refcount)
                CV_XADD(refcount, 1);
        }

        inline aclMat::aclMat(int _rows, int _cols, int _type, void *_data, aclCxt *_acl_context, ALIGNMENT config, size_t _step)
            : flags(0), rows(0), cols(0), step(0), data(nullptr), refcount(nullptr), datastart(nullptr),
              dataend(nullptr), offset(0), wholerows(0), wholecols(0), acl_context(_acl_context), totalSize(0)
        {
            cv::Mat m(_rows, _cols, _type, _data, _step);
            if (m.rows > 0 && m.cols > 0)
                create(m.rows, m.cols, m.type(), config);
            upload(m, config);
        }

        inline aclMat::aclMat(Size _size, int _type, void *_data, aclCxt *_acl_context, ALIGNMENT config, size_t _step)
            : flags(0), rows(0), cols(0), step(0), data(nullptr), refcount(nullptr), datastart(nullptr),
              dataend(nullptr), offset(0), wholerows(0), wholecols(0), acl_context(_acl_context), totalSize(0)
        {
            cv::Mat m(_size, _type, _data, _step);
            if (m.rows > 0 && m.cols > 0)
                create(m.rows, m.cols, m.type(), config);
            upload(m, config);
        }

        /**
         * @param [in] rRange: rows begin to end, Range(start, end)
         * @param [in] cRange: cols begin to end, Range(start, end) 
         */
        inline aclMat::aclMat(const aclMat &m, const Range &rRange, const Range &cRange)
            : flags(m.flags), step(m.step), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend),
              offset(m.offset), wholerows(m.wholerows), wholecols(m.wholecols), acl_context(m.acl_context), totalSize(m.totalSize)
        {
            if (rRange == Range::all())
                rows = m.rows;
            else
            {
                CV_Assert(0 <= rRange.start && rRange.start <= rRange.end && rRange.end <= m.rows);
                rows = rRange.size();
                offset += step * rRange.start;
            }

            if (cRange == Range::all())
                cols = m.cols;
            else
            {
                CV_Assert(0 <= cRange.start && cRange.start <= cRange.end && cRange.end <= m.cols);
                cols = cRange.size();
                offset += cRange.start * elemSize();
                flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
            }

            if (rows == 1)
                flags |= Mat::CONTINUOUS_FLAG;

            if (refcount)
                CV_XADD(refcount, 1);
            if (rows <= 0 || cols <= 0)
                rows = cols = 0;

            data = static_cast<void *>((static_cast<uchar *>(m.data) + offset));
        }

        /**
         *  @param [in] roi: Matrix position, Rect(x, y, width, height)
         *
         */
        inline aclMat::aclMat(const aclMat &m, const Rect &roi)
            : flags(m.flags), rows(roi.height), cols(roi.width), step(m.step), refcount(m.refcount), datastart(m.datastart),
              dataend(m.dataend), offset(m.offset), wholerows(m.wholerows), wholecols(m.wholecols), acl_context(m.acl_context),
              totalSize(m.totalSize)
        {
            flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
            offset += roi.y * step + roi.x * elemSize();
            CV_Assert(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.wholecols &&
                      0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.wholerows);
            if (refcount)
                CV_XADD(refcount, 1);
            if (rows <= 0 || cols <= 0)
                rows = cols = 0;

            data = static_cast<void *>((static_cast<uchar *>(m.data) + offset));
        }

        inline aclMat::aclMat(const Mat &m, aclCxt *_acl_context, ALIGNMENT config, MemMallocPolicy policy)
            : flags(0), rows(m.rows), cols(m.cols), step(0), data(nullptr), refcount(nullptr), datastart(nullptr),
              dataend(nullptr), offset(0), wholerows(0), wholecols(0), acl_context(_acl_context), totalSize(0)
        {
            if (m.rows > 0 && m.cols > 0)
                create(m.rows, m.cols, m.type(), config, policy);
            upload(m, config);
        }

        inline aclMat::~aclMat()
        {
            if (refcount)
                release();
        }

        inline aclMat &aclMat::operator=(const aclMat &m)
        {
            if (this != &m)
            {
                
                if (refcount)
                    release();
                flags = m.flags;
                rows = m.rows;
                cols = m.cols;
                step = m.step;
                datastart = m.datastart;
                dataend = m.dataend;
                offset = m.offset;
                wholerows = m.wholerows;
                wholecols = m.wholecols;
                refcount = m.refcount;
                acl_context = m.acl_context;
                totalSize = m.totalSize;
                data = m.data;
                if (m.refcount)
                    CV_XADD(m.refcount, 1);
            }
            return *this;
        }

        inline aclMat& aclMat::operator=(const Mat &m)
        {
            upload(m);
            return *this;
        }

        inline aclMat::operator Mat() const
        {
            Mat m(rows, cols, type());
            download(m);
            return m;
        }

        inline aclMat aclMat::clone() const
        {
            aclMat m;
            copyTo(m);
            return m;
        }

        inline void aclMat::copyTo(aclMat& dest) const
        {
            if (this != &dest)
            {
                dest.rows = rows;
                dest.cols = cols;
                dest.step = step;
                dest.wholerows = wholerows;
                dest.wholecols = wholecols;
                dest.refcount = refcount;
                dest.acl_context = acl_context;
                dest.totalSize = totalSize;

                void *dev_ptr;
                AclSafeCall(aclrtMalloc(&dev_ptr, totalSize, type_transition(MALLOC_NORMAL_ONLY)));
                AclSafeCall(aclrtMemcpy(dev_ptr, totalSize, data, totalSize, ACL_MEMCPY_DEVICE_TO_DEVICE));

                dest.data = dev_ptr;
                dest.datastart = static_cast<uchar *>(data);
                dest.dataend = static_cast<uchar *>(data) + totalSize;
                dest.refcount = static_cast<int *>(fastMalloc(sizeof(*refcount)));
                *refcount = 0;
                CV_XADD(refcount, 1);
                dest.flags |= Mat::CONTINUOUS_FLAG;
            }
        }

        inline aclMat aclMat::row(int y) const
        {
            return aclMat(*this, Range(y, y + 1), Range::all());
        }

        inline aclMat aclMat::col(int x) const
        {
            return aclMat(*this, Range::all(), Range(x, x + 1));
        }

        inline aclMat aclMat::rowRange(int startrow, int endrow) const
        {
            return aclMat(*this, Range(startrow, endrow), Range::all());
        }

        inline aclMat aclMat::rowRange(const Range &r) const
        {
            return aclMat(*this, r, Range::all());
        }

        inline aclMat aclMat::colRange(int startcol, int endcol) const
        {
            return aclMat(*this, Range::all(), Range(startcol, endcol));
        }

        inline aclMat aclMat::colRange(const Range &r) const
        {
            return aclMat(*this, Range::all(), r);
        }

        inline void aclMat::locateROI( Size &wholeSize, Point &ofs ) const
        {
            size_t esz = elemSize();
            CV_DbgAssert(step > 0);
            if(offset == 0)
                ofs.x = ofs.y = 0;
            else
            {
                ofs.y = (int)(offset / step);
                ofs.x = (int)((offset - step * ofs.y) / esz);
                CV_DbgAssert(data == (datastart + ofs.y * step + ofs.x * esz));
            }
            wholeSize.height = wholerows;
            wholeSize.width = wholecols;
        }

        inline aclMat &aclMat::adjustROI( int dtop, int dbottom, int dleft, int dright )
        {
            Size wholeSize;
            Point ofs;
            size_t esz = elemSize();
            locateROI( wholeSize, ofs );
            int row1 = std::max(ofs.y - dtop, 0), row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
            int col1 = std::max(ofs.x - dleft, 0), col2 = std::min(ofs.x + cols + dright, wholeSize.width);
            offset += (row1 - ofs.y) * step + (col1 - ofs.x) * esz;
            rows = row2 - row1;
            cols = col2 - col1;
            if( esz * cols == step || rows == 1 )
                flags |= Mat::CONTINUOUS_FLAG;
            else
                flags &= ~Mat::CONTINUOUS_FLAG;
            
            data = static_cast<void *>((static_cast<uchar *>(datastart) + offset));
            return *this;
        }

        inline void aclMat::swap(aclMat &b)
        {
            std::swap( flags, b.flags );
            std::swap( rows, b.rows );
            std::swap( cols, b.cols );
            std::swap( step, b.step );
            std::swap( data, b.data );
            std::swap( datastart, b.datastart );
            std::swap( dataend, b.dataend );
            std::swap( refcount, b.refcount );
            std::swap( offset, b.offset );
            std::swap( wholerows, b.wholerows );
            std::swap( wholecols, b.wholecols );
            std::swap( acl_context, b.acl_context);
            std::swap( totalSize, b.totalSize);
        }

        inline aclMat aclMat::operator()( Range rRange, Range cRange ) const
        {
            return aclMat(*this, rRange, cRange);
        }

        inline aclMat aclMat::operator()( const Rect &roi ) const
        {
            return aclMat(*this, roi);
        }

        inline bool aclMat::isContinuous() const
        {
            return (flags & Mat::CONTINUOUS_FLAG) != 0;
        }

        inline size_t aclMat::elemSize() const
        {
            return CV_ELEM_SIZE((CV_MAKE_TYPE(type(), channels())));
        }

        inline size_t aclMat::elemSize1() const
        {
            return CV_ELEM_SIZE1(flags);
        }

        inline int aclMat::type() const
        {
            return CV_MAT_TYPE(flags);
        }

        inline int aclMat::acltype() const
        {
            return CV_MAKE_TYPE(depth(), aclchannels());
        }

        inline int aclMat::depth() const
        {
            return CV_MAT_DEPTH(flags);
        }

        inline int aclMat::channels() const
        {
            return CV_MAT_CN(flags);
        }

        inline int aclMat::aclchannels() const
        {
            return (CV_MAT_CN(flags)) == 3 ? 4 : (CV_MAT_CN(flags));
        }

        inline size_t aclMat::step1() const
        {
            return step / elemSize1();
        }

        inline Size aclMat::size() const
        {
            return Size(cols, rows);
        }

        inline bool aclMat::empty() const
        {
            return data == 0;
        }

        inline void swap( aclMat &a, aclMat &b )
        {
            a.swap(b);
        }

        inline void ensureSizeIsEnough(int rows, int cols, int type, aclMat &m, ALIGNMENT config)
        {
            if (m.type() == type && m.rows >= rows && m.cols >= cols)
                m = m(Rect(0, 0, cols, rows));
            else
                m.create(rows, cols, type, config);
        }

        inline void ensureSizeIsEnough(Size size, int type, ALIGNMENT config, aclMat &m)
        {
            ensureSizeIsEnough(size.height, size.width, type, m, config);
        }

    } /* end of namespace acl */

} /* end of namespace cv */

#endif