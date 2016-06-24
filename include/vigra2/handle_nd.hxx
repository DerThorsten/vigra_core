/************************************************************************/
/*                                                                      */
/*               Copyright 2014-2016 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA2 computer vision library.          */
/*    The VIGRA2 Website is                                             */
/*        http://ukoethe.github.io/vigra2                               */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#pragma once

#ifndef VIGRA2_HANDLE_ND_HXX
#define VIGRA2_HANDLE_ND_HXX

#include <vector>
#include <utility>
#include "config.hxx"
#include "concepts.hxx"
#include "tinyarray.hxx"
#include "shape.hxx"

// Bounds checking Macro used if VIGRA_CHECK_BOUNDS is defined.
#ifdef VIGRA_CHECK_BOUNDS
#define VIGRA_ASSERT_INSIDE(diff) \
  vigra_precondition(this->isInside(diff), "Index out of bounds")
#else
#define VIGRA_ASSERT_INSIDE(diff)
#endif

namespace vigra {

template <int N, class T>
class HandleND
: public HandleNDTag
{
  public:
    static const int dimension                  = N;
    static const int shape_dimension            = N;
    typedef T                                     value_type;
    typedef typename std::add_const<T>::type      const_value_type;
    typedef value_type                          & reference;
    typedef const_value_type                    & const_reference;
    typedef typename std::remove_const<T>::type * pointer;
    typedef const_value_type                    * const_pointer;
    typedef Shape<shape_dimension>                difference_type;
    typedef ArrayIndex                            difference_type_1;
    typedef reference                             result_type;

    difference_type strides_;
    pointer         data_;

    HandleND(difference_type const & strides, const_pointer data)
    : strides_(strides)
    , data_(const_cast<pointer>(data))
    {}

        // length of consecutive array starting from 'axis'
    template <class SHAPE>
    ArrayIndex isConsecutive(SHAPE const & shape, int axis) const
    {
        ArrayIndex size = 1;
        for(int k=ndim()-1; k >= axis; --k)
        {
            if(size != strides_[k])
                return 0;
            size *= shape[k];
        }
        return size;
    }

        // only apply when array is consecutive!
    void inc() const
    {
        ++const_cast<pointer>(data_);
    }

        // only apply when array is consecutive!
    void dec() const
    {
        --const_cast<pointer>(data_);
    }

        // only apply when array is consecutive!
    void move(difference_type_1 diff) const
    {
        const_cast<pointer>(data_) += diff;
    }

    void inc(int axis) const
    {
        const_cast<pointer>(data_) += strides_[axis];
    }

    void dec(int axis) const
    {
        const_cast<pointer>(data_) -= strides_[axis];
    }

    void move(int axis, difference_type_1 diff) const
    {
        const_cast<pointer>(data_) += strides_[axis]*diff;
    }

    void move(difference_type const & diff) const
    {
        const_cast<pointer>(data_) += dot(strides_, diff);
    }

    reference operator*()
    {
        return *data_;
    }

    const_reference operator*() const
    {
        return *data_;
    }

    reference operator[](difference_type const & index)
    {
        return data_[dot(index, strides_)];
    }

    const_reference operator[](difference_type const & index) const
    {
        return data_[dot(index, strides_)];
    }

    pointer ptr()
    {
        return data_;
    }

    const_pointer ptr() const
    {
        return data_;
    }

    template <int M = N>
    int ndim(enable_if_t<M == runtime_size, bool> = true) const
    {
        return strides_.size();
    }

    template <int M = N>
    constexpr
    int ndim(enable_if_t<M != runtime_size, bool> = true) const
    {
        return N;
    }

    difference_type const & strides() const
    {
        return strides_;
    }
};

template <class T>
class HandleND<0, T>
: public HandleNDTag
{
  public:
    static const int dimension                  = 0;
    static const int shape_dimension            = runtime_size;
    typedef T                                     value_type;
    typedef typename std::add_const<T>::type      const_value_type;
    typedef value_type                          & reference;
    typedef const_value_type                    & const_reference;
    typedef value_type                          * pointer;
    typedef const_value_type                    * const_pointer;
    typedef Shape<shape_dimension>                difference_type;
    typedef ArrayIndex                            difference_type_1;
    typedef const_reference                       result_type;

    T data_;

    HandleND(T const & data)
    : data_(data)
    {}

        // length of consecutive array starting from 'axis'
    template <class SHAPE>
    ArrayIndex isConsecutive(SHAPE const & s, int dim) const
    {
        ArrayIndex res = 1;
        for(int k=dim; k<s.size(); ++k)
            res *= s[k];
        return res;
    }

    constexpr bool noMemoryOverlap(char *, char *) const
    {
        return true;
    }

    template <class SHAPE>
    constexpr bool compatibleMemoryLayout(char *, SHAPE const &) const
    {
        return true;
    }

    void inc() const {}
    void dec() const {}
    void move(difference_type_1) const {}
    void inc(int) const {}
    void dec(int) const {}
    void move(int, difference_type_1) const {}

    result_type operator*() const
    {
        return data_;
    }

    result_type operator[](difference_type const &) const
    {
        return data_;
    }

    pointer ptr()
    {
        return &data_;
    }

    const_pointer ptr() const
    {
        return &data_;
    }

    constexpr int ndim() const
    {
        return 0;
    }
};


  /**
     Handle class, used by CoupledScanOrderIterator as the value type to simultaneously itearate over multiple images.
  */
template <class T, class NEXT=void>
class HandleNDChain;
#if 0
: public NEXT
{
public:
    typedef NEXT                            base_type;
    typedef CoupledHandle<T, NEXT>          self_type;

    static const int index =                NEXT::index + 1;    // index of this member of the chain
    static const unsigned int dimensions =  NEXT::dimensions;

    typedef T                               value_type;
    typedef T *                             pointer;
    typedef T const *                       const_pointer;
    typedef T &                             reference;
    typedef T const &                       const_reference;
    typedef typename base_type::shape_type  shape_type;

    CoupledHandle()
    : base_type(),
      pointer_(),
      strides_()
    {}

    template <class NEXT1>
    CoupledHandle(CoupledHandle<T, NEXT1> const & h, NEXT const & next)
    : base_type(next),
      pointer_(h.pointer_),
      strides_(h.strides_)
    {}

    CoupledHandle(const_pointer p, shape_type const & strides, NEXT const & next)
    : base_type(next),
      pointer_(const_cast<pointer>(p)),
      strides_(strides)
    {}

    template <class Stride>
    CoupledHandle(MultiArrayView<dimensions, T, Stride> const & v, NEXT const & next)
    : base_type(next),
      pointer_(const_cast<pointer>(v.data())),
      strides_(v.stride())
    {
        vigra_precondition(v.shape() == this->shape(), "createCoupledIterator(): shape mismatch.");
    }

    inline void incDim(int dim)
    {
        pointer_ += strides_[dim];
        base_type::incDim(dim);
    }

    inline void decDim(int dim)
    {
        pointer_ -= strides_[dim];
        base_type::decDim(dim);
    }

    inline void addDim(int dim, MultiArrayIndex d)
    {
        pointer_ += d*strides_[dim];
        base_type::addDim(dim, d);
    }

    inline void add(shape_type const & d)
    {
        pointer_ += dot(d, strides_);
        base_type::add(d);
    }

    template<int DIMENSION>
    inline void increment()
    {
        pointer_ += strides_[DIMENSION];
        base_type::template increment<DIMENSION>();
    }

    template<int DIMENSION>
    inline void decrement()
    {
        pointer_ -= strides_[DIMENSION];
        base_type::template decrement<DIMENSION>();
    }

    // TODO: test if making the above a default case of the this hurts performance
    template<int DIMENSION>
    inline void increment(MultiArrayIndex offset)
    {
        pointer_ += offset*strides_[DIMENSION];
        base_type::template increment<DIMENSION>(offset);
    }

    template<int DIMENSION>
    inline void decrement(MultiArrayIndex offset)
    {
        pointer_ -= offset*strides_[DIMENSION];
        base_type::template decrement<DIMENSION>(offset);
    }

    void restrictToSubarray(shape_type const & start, shape_type const & end)
    {
        pointer_ += dot(start, strides_);
        base_type::restrictToSubarray(start, end);
    }

    // ptr access
    reference operator*()
    {
        return *pointer_;
    }

    const_reference operator*() const
    {
        return *pointer_;
    }

    pointer operator->()
    {
        return pointer_;
    }

    const_pointer operator->() const
    {
        return pointer_;
    }

    pointer ptr()
    {
        return pointer_;
    }

    const_pointer ptr() const
    {
        return pointer_;
    }

    shape_type const & strides() const
    {
        return strides_;
    }

    MultiArrayView<dimensions, T>
    arrayView() const
    {
        return MultiArrayView<dimensions, T>(this->shape(), strides(), ptr() - dot(this->point(), strides()));
    }

    template <unsigned int TARGET_INDEX>
    typename CoupledHandleCast<TARGET_INDEX, CoupledHandle, index>::reference
    get()
    {
        return vigra::get<TARGET_INDEX>(*this);
    }

    template <unsigned int TARGET_INDEX>
    typename CoupledHandleCast<TARGET_INDEX, CoupledHandle, index>::const_reference
    get() const
    {
        return vigra::get<TARGET_INDEX>(*this);
    }

    // NOTE: dangerous function - only use it when you know what you are doing
    void internal_reset(const_pointer p)
    {
        pointer_ = const_cast<pointer>(p);
    }

    pointer pointer_;
    shape_type strides_;
};
#endif

template <int N>
class HandleNDChain<Shape<N>, void>
{
public:
    static const unsigned int index      = 0; // index of this member of the chain
    static const unsigned int dimensions = N;

    typedef Shape<N>                       value_type;
    typedef value_type const *             pointer;
    typedef value_type const *             const_pointer;
    typedef value_type const &             reference;
    typedef value_type const &             const_reference;
    typedef value_type                     shape_type;
    typedef value_type                     difference_type;
    typedef HandleNDChain<Shape<N>, void>  self_type;

    HandleNDChain() = default;

    template <class SHAPE,
              VIGRA_REQUIRE<std::is_convertible<SHAPE, value_type>::value> >
    explicit HandleNDChain(SHAPE const & shape)
    : point_(tags::size = shape.size())
    , shape_(shape)
    {}

    // HandleNDChain(typename MultiArrayShape<N+1>::type const & shape)
    // : point_(),
      // shape_(shape.begin()),
      // scanOrderIndex_()
    // {}

    inline void inc(int dim)
    {
        ++point_[dim];
    }

    inline void dec(int dim)
    {
        --point_[dim];
    }

    void move(int dim, ArrayIndex diff)
    {
        point_[dim] += diff;
    }

    void move(difference_type const & diff)
    {
        point_ += diff;
    }

    // void restrictToSubarray(shape_type const & start, shape_type const & end)
    // {
        // point_ = shape_type();
        // shape_ = end - start;
    // }

    const_reference coord() const
    {
        return point_;
    }

    ArrayIndex coord(int dim) const
    {
        return point_[dim];
    }

    const_reference shape() const
    {
        return shape_;
    }

    ArrayIndex shape(int dim) const
    {
        return shape_[dim];
    }

    const_reference operator*() const
    {
        return point_;
    }

    const_pointer operator->() const
    {
        return &point_;
    }

    template <class SHAPE,
              VIGRA_REQUIRE<std::is_convertible<SHAPE, value_type>::value> >
    value_type operator[](SHAPE const & diff) const
    {
        return point_ + diff;
    }

    const_pointer ptr() const
    {
        return &point_;
    }

    template <int M = N>
    int ndim(enable_if_t<M == runtime_size, bool> = true) const
    {
        return shape_.size();
    }

    template <int M = N>
    constexpr
    int ndim(enable_if_t<M != runtime_size, bool> = true) const
    {
        return N;
    }

    // unsigned int borderType() const
    // {
        // return detail::BorderTypeImpl<N>::exec(point_, shape_);
    // }

    // template <unsigned int TARGET_INDEX>
    // typename CoupledHandleCast<TARGET_INDEX, CoupledHandle, index>::reference
    // get()
    // {
        // return vigra::get<TARGET_INDEX>(*this);
    // }

    // template <unsigned int TARGET_INDEX>
    // typename CoupledHandleCast<TARGET_INDEX, CoupledHandle, index>::const_reference
    // get() const
    // {
        // return vigra::get<TARGET_INDEX>(*this);
    // }

    value_type point_, shape_;
};

namespace array_detail {

template <class SHAPE>
inline SHAPE
permutationToOrder(SHAPE const & shape, SHAPE const & stride,
                   MemoryOrder order)
{
    using V = typename SHAPE::value_type;
    SHAPE res = SHAPE::range(shape.size());
    if(order == C_ORDER)
        std::sort(res.begin(), res.end(),
                 [shape, stride](V l, V r)
                 {
                    if(shape[l] == 1 || shape[r] == 1)
                        return shape[l] < shape[r];
                    return stride[r] < stride[l];
                 });
    else
        std::sort(res.begin(), res.end(),
                 [shape, stride](V l, V r)
                 {
                    if(shape[l] == 1 || shape[r] == 1)
                        return shape[r] < shape[l];
                    return stride[l] < stride[r];
                 });
    return res;
}

template <class HANDLE, class SHAPE, class FCT,
          VIGRA_REQUIRE<HandleNDConcept<HANDLE>::value> >
void
genericArrayFunctionImpl(HANDLE & h, SHAPE const & shape, FCT f, int dim = 0)
{
    vigra_assert(dim < shape.size(),
        "genericArrayFunctionImpl(): internal error: dim >= shape.size() should never happen.");

    auto N = h.isConsecutive(shape, dim);
    if(N)
    {
        auto p = h.ptr();
        for(ArrayIndex k=0; k<N; ++k, ++p)
            f(*p);
    }
    else
    {
        N = shape[dim];
        if(dim == shape.size() - 1)
        {
            for(ArrayIndex k=0; k<N; ++k, h.inc(dim))
                f(*h);
        }
        else
        {
            for(ArrayIndex k=0; k<N; ++k, h.inc(dim))
                genericArrayFunctionImpl(h, shape, f, dim+1);
        }
        h.move(dim, -N);
    }
}

template <class ARRAY, class FCT,
          VIGRA_REQUIRE<ArrayNDConcept<ARRAY>::value> >
void
genericArrayFunction(ARRAY & a, FCT f)
{
    auto p = permutationToOrder(a.shape(), a.strides(), C_ORDER);
    auto h = a.handle(p);
    auto s = transpose(a.shape(), p);
    genericArrayFunctionImpl(h, s, f);
}

template <class HANDLE1, class HANDLE2, class SHAPE, class FCT,
          VIGRA_REQUIRE<HandleNDConcept<HANDLE1>::value && HandleNDConcept<HANDLE2>::value> >
void
genericArrayFunctionImpl(HANDLE1 & h1, HANDLE2 & h2, SHAPE const & shape,
                         FCT f, int dim = 0)
{
    vigra_assert(dim < shape.size(),
        "genericArrayFunctionImpl(): internal error: dim >= shape.size() should never happen.");
    auto N = h1.isConsecutive(shape, dim);
    if(N && N == h2.isConsecutive(shape, dim))
    {
        auto p1 = h1.ptr();
        auto p2 = h2.ptr();
        for(ArrayIndex k=0; k<N; ++k, ++p1, ++p2)
            f(*p1, *p2);
    }
    else
    {
        N = shape[dim];
        if(dim == shape.size() - 1)
        {
            for(ArrayIndex k=0; k<N; ++k, h1.inc(dim), h2.inc(dim))
                f(*h1, *h2);
        }
        else
        {
            for(ArrayIndex k=0; k<N; ++k, h1.inc(dim), h2.inc(dim))
                genericArrayFunctionImpl(h1, h2, shape, f, dim+1);
        }
        h1.move(dim, -N);
        h2.move(dim, -N);
    }
}

template <class ARRAY1, class ARRAY2, class FCT>
enable_if_t<ArrayNDConcept<ARRAY1>::value && ArrayNDConcept<ARRAY2>::value>
genericArrayFunction(ARRAY1 & a1, ARRAY2 const & a2, FCT f)
{
    auto last = a1.shape() - 1;
    char * p1 = (char *)a1.data();
    char * q1 = (char *)(&a1[last]+1);
    char * p2 = (char *)a2.data();
    char * q2 = (char *)(&a2[last]+1);

    bool no_overlap        = q1 <= p2 || q2 <= p1;
    bool compatible_layout = p1 <= p2 && a1.strides() == a2.strides();

    auto p  = permutationToOrder(a1.shape(), a1.strides(), C_ORDER);
    auto h1 = a1.handle(p);
    auto s  = transpose(a1.shape(), p);

    if(no_overlap || compatible_layout)
    {
        auto h2 = a2.handle(p);
        genericArrayFunctionImpl(h1, h2, s, f);
    }
    else
    {
        using TmpArray = ArrayND<ARRAY2::dimension, typename ARRAY2::value_type>;
        TmpArray t2(a2);
        auto h2 = t2.handle(p);
        genericArrayFunctionImpl(h1, h2, s, f);
    }
}

    // if both arrays are read-only, we need not worry about overlapping memory
template <class ARRAY1, class ARRAY2, class FCT>
enable_if_t<ArrayNDConcept<ARRAY1>::value && ArrayNDConcept<ARRAY2>::value>
genericArrayFunction(ARRAY1 const & a1, ARRAY2 const & a2, FCT f)
{
    auto p  = permutationToOrder(a1.shape(), a1.strides(), C_ORDER);
    auto h1 = a1.handle(p);
    auto h2 = a2.handle(p);
    auto s  = transpose(a1.shape(), p);

    genericArrayFunctionImpl(h1, h2, s, f);
}

template <class ARRAY1, class ARRAY2, class FCT>
enable_if_t<ArrayNDConcept<ARRAY1>::value && ArrayMathConcept<ARRAY2>::value>
genericArrayFunction(ARRAY1 & a1, ARRAY2 const & h2, FCT f)
{
    auto last = a1.shape() - 1;
    char * p1 = (char *)a1.data();
    char * q1 = (char *)(&a1[last]+1);

    bool no_overlap        = h2.noMemoryOverlap(p1, q1);
    bool compatible_layout = h2.compatibleMemoryLayout(p1, a1.strides());

    auto p  = permutationToOrder(a1.shape(), a1.strides(), C_ORDER);
    auto h1 = a1.handle(p);
    auto s  = transpose(a1.shape(), p);

    if(no_overlap || compatible_layout)
    {
        h2.transpose(p);
        genericArrayFunctionImpl(h1, h2, s, f);
    }
    else
    {
        using TmpArray = ArrayND<ARRAY1::dimension, typename ARRAY2::value_type>;
        TmpArray t2(h2);
        auto ht = t2.handle(p);
        genericArrayFunctionImpl(h1, ht, s, f);
    }
}

} // namespace array_detail

} // namespace vigra

#endif // VIGRA2_HANDLE_ND_HXX
