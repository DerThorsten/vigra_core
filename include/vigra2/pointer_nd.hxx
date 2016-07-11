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

#ifndef VIGRA2_POINTER_ND_HXX
#define VIGRA2_POINTER_ND_HXX

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
class ArrayViewND;

template <int N, class T, class Alloc = std::allocator<T> >
class ArrayND;

    // The PointerND class combines a pointer to an array element with a stride
    // object, so that the appropriate N-D pointer arithmetic can be implemented.
    //
    // The specialization PointerND<0, T> further down below is used for constants
    // and implements all pointer arithmetic as no-ops.
template <int N, class T>
class PointerND
: public PointerNDTag
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

    PointerND(difference_type const & strides, const_pointer data)
    : strides_(strides)
    , data_(const_cast<pointer>(data))
    {}

        // length of consecutive array starting from 'axis'
        //
        // This function always assumes C-order!
        // FIXME: should isConsecutive() be moved from PointerND to another class?
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

    bool hasData() const
    {
        return data_ != 0;
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
class PointerND<0, T>
: public PointerNDTag
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

    PointerND(T const & data)
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

    constexpr bool hasData() const
    {
        return true;
    }

    void inc() const {}
    void dec() const {}
    void move(difference_type_1) const {}
    void inc(int) const {}
    void dec(int) const {}
    void move(int, difference_type_1) const {}
    void move(difference_type const &) const {}

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
     Coupled PointerND objects are used by IteratorND to itearate over multiple arrays simultaneously.
  */
template <class T, class NEXT = void>
class PointerNDCoupled
: public NEXT
{
  public:
    typedef NEXT                          base_type;
    static const unsigned int index     = base_type::index+1; // index of this member of the chain
    static const unsigned int dimension = base_type::dimension;
    typedef PointerND<dimension, T>       pointer_nd_type;

    typedef typename pointer_nd_type::value_type       value_type;
    typedef typename pointer_nd_type::pointer          pointer;
    typedef typename pointer_nd_type::const_pointer    const_pointer;
    typedef typename pointer_nd_type::reference        reference;
    typedef typename pointer_nd_type::const_reference  const_reference;
    typedef typename base_type::shape_type             shape_type;
    typedef typename base_type::difference_type        difference_type;
    typedef PointerNDCoupled                           self_type;

    PointerNDCoupled() = default;

    PointerNDCoupled(PointerND<dimension, T> const & pointer_nd, NEXT const & next)
    : base_type(next)
    , pointer_nd_(pointer_nd)
    {}

    inline void inc(int dim)
    {
        base_type::inc(dim);
        pointer_nd_.inc(dim);
    }

    inline void dec(int dim)
    {
        base_type::dec(dim);
        pointer_nd_.dec(dim);
    }

    void move(int dim, ArrayIndex diff)
    {
        base_type::move(dim, diff);
        pointer_nd_.move(dim, diff);
    }

    void move(difference_type const & diff)
    {
        base_type::move(diff);
        pointer_nd_.move(diff);
    }

    // void restrictToSubarray(shape_type const & start, shape_type const & end)
    // {
        // point_ = shape_type();
        // shape_ = end - start;
    // }

    reference operator*()
    {
        return *pointer_nd_;
    }

    const_reference operator*() const
    {
        return *pointer_nd_;
    }

    pointer operator->()
    {
        return pointer_nd_.ptr();
    }

    const_pointer operator->() const
    {
        return pointer_nd_.ptr();
    }

    value_type operator[](shape_type const & diff) const
    {
        return pointer_nd_[diff];
    }

    const_pointer ptr() const
    {
        return pointer_nd_.ptr();
    }

    pointer_nd_type pointer_nd_;
};

template <int N>
class PointerNDCoupled<Shape<N>, void>
{
public:
    static const unsigned int index      = 0; // index of this member of the chain
    static const unsigned int dimension  = N;

    typedef Shape<N>                       value_type;
    typedef value_type const *             pointer;
    typedef value_type const *             const_pointer;
    typedef value_type const &             reference;
    typedef value_type const &             const_reference;
    typedef value_type                     shape_type;
    typedef value_type                     difference_type;
    typedef PointerNDCoupled               self_type;
    typedef const_reference                result_type;

    PointerNDCoupled() = default;

    explicit PointerNDCoupled(shape_type const & shape)
    : point_(tags::size = shape.size())
    , shape_(shape)
    {}

    template <class SHAPE>
    constexpr ArrayIndex isConsecutive(SHAPE const &, int) const
    {
        return 0;
    }

    inline void inc()
    {
        vigra_invariant(false,
            "PointerNDShape::inc(): not allowed because PointerNDCoupled has no consecutive memory.");
    }

    inline void dec()
    {
        vigra_invariant(false,
            "PointerNDShape::dec(): not allowed because PointerNDCoupled has no consecutive memory.");
    }

    void move(ArrayIndex)
    {
        vigra_invariant(false,
            "PointerNDShape::move(ArrayIndex): not allowed because PointerNDCoupled has no consecutive memory.");
    }

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

    value_type operator[](shape_type const & diff) const
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

    value_type point_, shape_;
};

template <int N>
using PointerNDShape = PointerNDCoupled<Shape<N>, void>;

namespace array_detail {

template <class COUPLED_POINTERS, class ... REST>
struct PointerNDTypeImpl;

template <class COUPLED_POINTERS, class T, class ... REST>
struct PointerNDTypeImpl<COUPLED_POINTERS, T, REST...>
{
    typedef typename PointerNDTypeImpl<PointerNDCoupled<T, COUPLED_POINTERS>,
                                         REST...>::type    type;
};

template <class COUPLED_POINTERS, int N, class T, class ... REST>
struct PointerNDTypeImpl<COUPLED_POINTERS, ArrayViewND<N, T>, REST...>
{
    static_assert(CompatibleDimensions<N, COUPLED_POINTERS::dimension>::value,
        "PointerNDCoupled<...>: dimension mismatch.");
    typedef typename PointerNDTypeImpl<PointerNDCoupled<T, COUPLED_POINTERS>,
                                         REST...>::type    type;
};

template <class COUPLED_POINTERS, int N, class T, class A, class ... REST>
struct PointerNDTypeImpl<COUPLED_POINTERS, ArrayND<N, T, A>, REST...>
{
    static_assert(CompatibleDimensions<N, COUPLED_POINTERS::dimension>::value,
        "PointerNDCoupled<...>: dimension mismatch.");
    typedef typename PointerNDTypeImpl<PointerNDCoupled<T, COUPLED_POINTERS>,
                                         REST...>::type    type;
};

template <class T, class U>
struct PointerNDTypeImpl<PointerNDCoupled<T, U>>
{
    typedef PointerNDCoupled<T, U> type;
};

} // namespace array_detail

template <int N, class ... REST>
using PointerNDCoupledType = typename array_detail::PointerNDTypeImpl<PointerNDShape<N>, REST...>::type;

namespace array_detail {

template <int K, class COUPLED_POINTERS, bool MATCH = (K == COUPLED_POINTERS::index)>
struct PointerNDCoupledCast
{
    static_assert( 0 <= K && K < COUPLED_POINTERS::index,
        "get<INDEX>(): index out of range.");

    typedef PointerNDCoupledCast<K, typename COUPLED_POINTERS::base_type> Next;
    typedef typename Next::type type;

    static type & cast(COUPLED_POINTERS & h)
    {
        return Next::cast(h);
    }

    static type const & cast(COUPLED_POINTERS const & h)
    {
        return Next::cast(h);
    }
};

template <int K, class COUPLED_POINTERS>
struct PointerNDCoupledCast<K, COUPLED_POINTERS, true>
{
    typedef COUPLED_POINTERS type;

    static type & cast(COUPLED_POINTERS & h)
    {
        return h;
    }

    static type const & cast(COUPLED_POINTERS const & h)
    {
        return h;
    }
};

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

template <class POINTER_ND, class SHAPE, class FCT,
          VIGRA_REQUIRE<PointerNDConcept<POINTER_ND>::value> >
void
universalPointerNDFunction(POINTER_ND & h, SHAPE const & shape, FCT f, int dim = 0)
{
    vigra_assert(dim < shape.size(),
        "universalPointerNDFunction(): internal error: dim >= shape.size() should never happen.");

    auto N = h.isConsecutive(shape, dim);
    if(N)
    {
        // auto p = h.ptr();
        // for(ArrayIndex k=0; k<N; ++k, ++p)
            // f(*p);

            // Use h.inc() and *h to make the loop work for scalar pointer_nds
            // (who have zero stride) as well. To specialize for
            // low-level optimizations like AVX, we need to distinguish
            // if pointer_nds refer to arrays or scalars.
        for(ArrayIndex k=0; k<N; ++k, h.inc())
            f(*h);
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
                universalPointerNDFunction(h, shape, f, dim+1);
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
    auto h = a.pointer_nd(p);
    auto s = transpose(a.shape(), p);
    universalPointerNDFunction(h, s, f);
}

template <class POINTER_ND1, class POINTER_ND2, class SHAPE, class FCT,
          VIGRA_REQUIRE<PointerNDConcept<POINTER_ND1>::value && PointerNDConcept<POINTER_ND2>::value> >
void
universalPointerNDFunction(POINTER_ND1 & h1, POINTER_ND2 & h2, SHAPE const & shape,
                         FCT f, int dim = 0)
{
    vigra_assert(dim < shape.size(),
        "universalPointerNDFunction(): internal error: dim >= shape.size() should never happen.");
    auto N = h1.isConsecutive(shape, dim);
    if(N && N == h2.isConsecutive(shape, dim))
    {
        // auto p1 = h1.ptr();
        // auto p2 = h2.ptr();
        // for(ArrayIndex k=0; k<N; ++k, ++p1, ++p2)
            // f(*p1, *p2);

            // Use h1.inc() and *h1 to make the loop work for scalar pointer_nds
            // (who have zero stride) as well. To specialize for
            // low-level optimizations like AVX, we need to distinguish
            // if pointer_nds refer to arrays or scalars.
        for(ArrayIndex k=0; k<N; ++k, h1.inc(), h2.inc())
            f(*h1, *h2);
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
                universalPointerNDFunction(h1, h2, shape, f, dim+1);
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
    auto h1 = a1.pointer_nd(p);
    auto s  = transpose(a1.shape(), p);

    if(no_overlap || compatible_layout)
    {
        auto h2 = a2.pointer_nd(p);
        universalPointerNDFunction(h1, h2, s, f);
    }
    else
    {
        using TmpArray = ArrayND<ARRAY2::dimension, typename ARRAY2::value_type>;
        TmpArray t2(a2);
        auto h2 = t2.pointer_nd(p);
        universalPointerNDFunction(h1, h2, s, f);
    }
}

    // if both arrays are read-only, we need not worry about overlapping memory
template <class ARRAY1, class ARRAY2, class FCT>
enable_if_t<ArrayNDConcept<ARRAY1>::value && ArrayNDConcept<ARRAY2>::value>
genericArrayFunction(ARRAY1 const & a1, ARRAY2 const & a2, FCT f)
{
    auto p  = permutationToOrder(a1.shape(), a1.strides(), C_ORDER);
    auto h1 = a1.pointer_nd(p);
    auto h2 = a2.pointer_nd(p);
    auto s  = transpose(a1.shape(), p);

    universalPointerNDFunction(h1, h2, s, f);
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
    auto h1 = a1.pointer_nd(p);
    auto s  = transpose(a1.shape(), p);

    if(no_overlap || compatible_layout)
    {
        h2.transpose(p);
        universalPointerNDFunction(h1, h2, s, f);
    }
    else
    {
        using TmpArray = ArrayND<ARRAY1::dimension, typename ARRAY2::value_type>;
        TmpArray t2(h2);
        auto ht = t2.pointer_nd(p);
        universalPointerNDFunction(h1, ht, s, f);
    }
}

} // namespace array_detail


template <int INDEX, class T, class NEXT>
auto
get(PointerNDCoupled<T, NEXT> const & h)
-> decltype(*array_detail::PointerNDCoupledCast<INDEX, PointerNDCoupled<T, NEXT>>::cast(h))
{
    return *array_detail::PointerNDCoupledCast<INDEX, PointerNDCoupled<T, NEXT>>::cast(h);
}

template <int INDEX, class T, class NEXT>
auto
get(PointerNDCoupled<T, NEXT> & h)
-> decltype(*array_detail::PointerNDCoupledCast<INDEX, PointerNDCoupled<T, NEXT>>::cast(h))
{
    return *array_detail::PointerNDCoupledCast<INDEX, PointerNDCoupled<T, NEXT>>::cast(h);
}


} // namespace vigra

#endif // VIGRA2_POINTER_ND_HXX
