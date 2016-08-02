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

/* This header contains VIGRA's low-level APIs for multi-dimensional arrays.
*/

/********************************************************/
/*                                                      */
/*                  forward declarations                */
/*                                                      */
/********************************************************/

template <int N, class T>
class ArrayViewND;

template <int N, class T, class Alloc = std::allocator<T> >
class ArrayND;

namespace array_math {

// Forward declarations.
template <class, class>
struct ArrayMathUnifyDimension;

// Choose the appropriate ArrayMathExpression according to the ARG type.
template <class>
struct ArrayMathTypeChooser;

template <class>
struct ArrayMathExpression;

} // namespace array_math

/********************************************************/
/*                                                      */
/*                       PointerND                      */
/*                                                      */
/********************************************************/

    // The PointerND class provides a simple low-level API to deal with
    // multi-dimensional arrays. It combines a pointer to an array element
    // with a stride object and supports the corresponding N-dimensional
    // pointer arithmetic.
    //
    // The specialization PointerND<0, T> further down below is used for
    // 0-dimensional arrays (i.e. constants) and implements all pointer
    // arithmetic as no-ops.
    //
    // Note: Strides are internally stored in units of bytes, whereas the
    // external API always measures strides in units of `sizeof(T)`, unless
    // byte-strides are explicitly enforced by calling the `byte_strides()`
    // member function or passing strides via the `tags::byte_strides` keyword
    // argument.
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
    char * data_;

    PointerND()
    : strides_()
    , data_(0)
    {}

    PointerND(difference_type const & strides, const_pointer data)
    : strides_(strides*sizeof(T))
    , data_((char*)data)
    {}

    PointerND(tags::ByteStridesProxy<N> const & strides, const_pointer data)
    : strides_(strides.value)
    , data_((char*)data)
    {}

    bool hasData() const
    {
        return data_ != 0;
    }

    void inc(int axis)
    {
        data_ += strides_[axis];
    }

    void dec(int axis)
    {
        data_ -= strides_[axis];
    }

    void move(int axis, difference_type_1 diff)
    {
        data_ += strides_[axis]*diff;
    }

    void move(difference_type const & diff)
    {
        data_ += dot(strides_, diff);
    }

    reference operator*()
    {
        return *ptr();
    }

    const_reference operator*() const
    {
        return *ptr();
    }

    reference operator[](difference_type const & index)
    {
        return *(pointer)(data_ + dot(index, strides_));
    }

    const_reference operator[](difference_type const & index) const
    {
        return *(const_pointer)(data_ + dot(index, strides_));
    }

    pointer ptr()
    {
        return (pointer)data_;
    }

    const_pointer ptr() const
    {
        return (const_pointer)data_;
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

    difference_type const & byte_strides() const
    {
        return strides_;
    }

    ArrayIndex strides(int dim) const
    {
        return strides_[dim] / sizeof(T);
    }

    template <class STRIDES>
    bool
    compatibleStrides(STRIDES const & other) const
    {
        if (other.size() == 0)
            return true;
        // FIXME: can singletons be considered as compatible strides?
        return strides_ == other;
    }

    template <class SHAPE>
    PointerND
    pointer_nd(SHAPE const & permutation) const
    {
        return PointerND(tags::byte_strides = strides_.transpose(permutation), (T const*)data_);
    }
};

/********************************************************/
/*                                                      */
/*                    PointerND<0, T>                   */
/*                                                      */
/********************************************************/

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
    typedef const_reference                       result_type;

    T data_;

    PointerND(T const & data)
    : data_(data)
    {}

    constexpr bool hasData() const
    {
        return true;
    }

    void inc(int) const {}
    void dec(int) const {}
    void move(int, ArrayIndex) const {}
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

    constexpr ArrayIndex strides(int) const
    {
        return 0;
    }

    template <class STRIDES>
    constexpr bool compatibleStrides(STRIDES const &) const
    {
        return true;
    }

    template <class SHAPE>
    PointerND
    pointer_nd(SHAPE const &) const
    {
        return PointerND(*this);
    }
};


/********************************************************/
/*                                                      */
/*                    PointerNDCoupled                  */
/*                                                      */
/********************************************************/

  /*
     PointerNDCoupled combines several PointerND objects and a shape object into
     a linked list. The list supports the basic PointerND API and applies pointer arithmetic
     to all members of the list (i.e. all coupled arrays) synchronously. This functionality
     is, for example, used by IteratorND to itearate over multiple coupled arrays simultaneously.
  */
template <class T, class NEXT = void>
class PointerNDCoupled
: public NEXT
{
  public:
    typedef NEXT                          base_type;
    static const int index =              base_type::index+1; // index of this member of the chain
    static const int dimension =          base_type::dimension;
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

/********************************************************/
/*                                                      */
/*               PinterNDCoupled<Shape<N>>              */
/*                                                      */
/********************************************************/

    // PointerNDCoupled<Shape<N>> is the terminal member in a list of coupled
    // PointerND objects. It holds the shape object shared by those coupled
    // arrays along with the current coordinate of an iteration across this shape.
    // To manipulate this coordinate, is provides the basic PointerND API.
template <int N>
class PointerNDCoupled<Shape<N>>
{
public:
    static const int index =               0; // index of this member of the chain
    static const int dimension =           N;

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

    unsigned int borderType() const
    {
        unsigned int res = 0;
        for(int k=0; k<ndim(); ++k)
        {
            if(point_[k] == 0)
                res |= (1 << 2*k);
            if(point_[k] == shape_[k]-1)
                res |= (2 << 2*k);
        }
        return res;
    }

    value_type point_, shape_;
};

template <int N>
using PointerNDShape = PointerNDCoupled<Shape<N>>;

namespace array_detail {

/********************************************************/
/*                                                      */
/*                  PointerNDCoupledCast                */
/*                                                      */
/********************************************************/

    // helper classes to extract the elements of a PointerNDCoupled list.
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

/********************************************************/
/*                                                      */
/*                   PointerNDTypeImpl                  */
/*                                                      */
/********************************************************/

    // helper classes to construct PointerNDCoupled lists
template <class POINTERS, class ... REST>
struct PointerNDTypeImpl;

template <class POINTERS, class ARRAY, class ... REST>
struct PointerNDTypeImpl<POINTERS, ARRAY, REST...>
{
    static_assert(CompatibleDimensions<NDimTraits<ARRAY>::value, POINTERS::dimension>::value,
        "makePointerNDCoupled(): dimension mismatch.");
    typedef typename
        PointerNDTypeImpl<PointerNDCoupled<typename ValueTypeTraits<ARRAY>::type, POINTERS>,
                          REST...>::type
        type;
};

template <class T, class U>
struct PointerNDTypeImpl<PointerNDCoupled<T, U>>
{
    typedef PointerNDCoupled<T, U> type;
};

/********************************************************/
/*                                                      */
/*               makePointerNDCoupledImpl               */
/*                                                      */
/********************************************************/

    // factory helpers to construct PointerNDCoupled lists
template <class COUPLED_POINTERS>
inline COUPLED_POINTERS &&
makePointerNDCoupledImpl(COUPLED_POINTERS && pointers)
{
    return std::forward<COUPLED_POINTERS>(pointers);
}

template <class COUPLED_POINTERS, class ARRAY, class ... REST>
typename PointerNDTypeImpl<COUPLED_POINTERS, ARRAY, REST...>::type
makePointerNDCoupledImpl(COUPLED_POINTERS const & inner_pointers,
                         ARRAY && a, REST && ... rest)
{
    static_assert(ArrayNDConcept<ARRAY>::value,
        "makePointerNDCoupled(): arguments must fulfill the ArrayNDConcept.");

    typedef typename ValueTypeTraits<ARRAY>::type T;

    PointerNDCoupled<T, COUPLED_POINTERS> pointer_nd(a.pointer_nd(), inner_pointers);
    return makePointerNDCoupledImpl(pointer_nd, std::forward<REST>(rest) ...);
}

} // namespace array_detail

/********************************************************/
/*                                                      */
/*                 makePointerNDCoupled                 */
/*                                                      */
/********************************************************/

template <class ARRAY, class ... REST>
using PointerNDCoupledType = typename
    array_detail::PointerNDTypeImpl<PointerNDShape<NDimTraits<ARRAY>::value>, ARRAY, REST...>::type;

    // factory function to construct PointerNDCoupled lists
template <class ARRAY, class ... REST,
          VIGRA_REQUIRE<ArrayNDConcept<ARRAY>::value> >
PointerNDCoupledType<ARRAY, REST...>
makePointerNDCoupled(ARRAY && a, REST && ... rest)
{
    typedef typename ValueTypeTraits<ARRAY>::type T;
    static const int N = NDimTraits<ARRAY>::value;

    PointerNDCoupled<T, PointerNDShape<N>> pointer_nd(a.pointer_nd(),
                                                      PointerNDShape<N>(a.shape()));
    return array_detail::makePointerNDCoupledImpl(pointer_nd, std::forward<REST>(rest) ...);
}

/********************************************************/
/*                                                      */
/*            get<INDEX>(PointerNDCoupled)              */
/*                                                      */
/********************************************************/

    // extract the current element at `INDEX` during a coupled array iteration
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

 namespace array_detail {

/********************************************************/
/*                                                      */
/*                  checkMemoryOverlap                  */
/*                                                      */
/********************************************************/

    // check if and how the memory of two arrays overlaps

enum MemoryOverlap {
    NoMemoryOverlap = 0,
    TargetOverlapsLeft = 1,
    TargetOverlapsRight = 2,
    TargetOverlaps = 3
};

inline MemoryOverlap
checkMemoryOverlap(TinyArray<char*, 2> const & target, TinyArray<char*, 2> const & src)
{
    MemoryOverlap res = NoMemoryOverlap;
    if (target[1] <= src[0] || src[1] <= target[0])
        return res;
    if (target[0] <= src[0] || target[1] <= src[1])
        res = (MemoryOverlap)(res | TargetOverlapsLeft);
    if (src[0] <= target[0] || src[1] <= target[1])
        res = (MemoryOverlap)(res | TargetOverlapsRight);
    return res;
}

template <class ARRAY>
inline
enable_if_t<ArrayNDConcept<ARRAY>::value, MemoryOverlap>
checkMemoryOverlap(TinyArray<char*, 2> const & target, ARRAY const & src)
{
    return checkMemoryOverlap(target, src.memoryRange());
}

template <class ARRAY>
inline
enable_if_t<ArrayMathConcept<ARRAY>::value, MemoryOverlap>
checkMemoryOverlap(TinyArray<char*, 2> const & target, ARRAY const & src)
{
    return src.checkMemoryOverlap(target);
}

/********************************************************/
/*                                                      */
/*                      unifyShape                      */
/*                                                      */
/********************************************************/

    // Create a common shape for `src` and the old state of `target`,
    // where singleton axes are expanded to the size of the corresponding
    // axis in the other shape. The result is stored in `target`.
    // The function returns `false` if the shapes are incompatible.
template <int N, class ARRAY>
inline
enable_if_t<ArrayNDConcept<ARRAY>::value, bool>
unifyShape(Shape<N> & target, ARRAY const & src)
{
    return detail::unifyShape(target, src.shape());
}

template <int N, class ARRAY>
inline
enable_if_t<ArrayMathConcept<ARRAY>::value, bool>
unifyShape(Shape<N> & target, ARRAY const & src)
{
    return src.unifyShape(target);
}

/********************************************************/
/*                                                      */
/*                  principalStrides                    */
/*                                                      */
/********************************************************/

    // Figure out the strides of the most important (biggest) array in a set of arrays
template <class SHAPE1, class SHAPE2>
inline void
principalStrides(SHAPE1 & target, SHAPE2 const & src,
                 ArrayIndex & minimalStride, int & singletonCount)
{
    if (src.size() != target.size())
        return;
    ArrayIndex m = NumericTraits<ArrayIndex>::max();
    int        s = 0;
    for (int k = 0; k < src.size(); ++k)
    {
        if (src[k] == 0)
            ++s;
        else if (src[k] < m)
            m = src[k];
    }
    if (s <= singletonCount && m <= minimalStride)
    {
        singletonCount = s;
        minimalStride = m;
        target = src;
    }
}

template <int N, class ARRAY1, class ARRAY2>
inline
enable_if_t<ArrayNDConcept<ARRAY1>::value && ArrayNDConcept<ARRAY2>::value>
principalStrides(Shape<N> & strides, ARRAY1 const & a1, ARRAY2 const & a2)
{
    ArrayIndex minimalStride = NumericTraits<ArrayIndex>::max();
    int singletonCount = strides.size();
    principalStrides(strides, a1.byte_strides(), minimalStride, singletonCount);
    principalStrides(strides, a2.byte_strides(), minimalStride, singletonCount);
}

template <int N, class ARRAY1, class ARRAY2>
inline
enable_if_t<ArrayNDConcept<ARRAY1>::value && ArrayMathConcept<ARRAY2>::value>
principalStrides(Shape<N> & strides, ARRAY1 const & a1, ARRAY2 const & a2)
{
    ArrayIndex minimalStride = NumericTraits<ArrayIndex>::max();
    int singletonCount = strides.size();
    principalStrides(strides, a1.byte_strides(), minimalStride, singletonCount);
    a2.principalStrides(strides, minimalStride, singletonCount);
}

/********************************************************/
/*                                                      */
/*                    isCConsecutive                    */
/*                                                      */
/********************************************************/

   // Check if the given PointerND refers to C-order consecutive memory for the given shape
   // from 'dim' up to the last dimension (dimensions befor 'dim' don't matter).
   // Returns the number of consecutive elements if the test is successful, or zero otherwise.
template <int N, class T, class SHAPE>
inline ArrayIndex
isCConsecutive(PointerND<N, T> const & p, SHAPE const & shape, int dim)
{
    ArrayIndex size = sizeof(T);
    for(int k=shape.size()-1; k >= dim; --k)
    {
        if(size != p.byte_strides()[k])
            return 0;
        size *= shape[k];
    }
    return size / sizeof(T);
}

/********************************************************/
/*                                                      */
/*                  forwardPointerND                    */
/*                                                      */
/********************************************************/

    // Create an ArrayMathExpression from a PointerND and shape.
    // Just forwards `src` if it already is an ArrayMathExpression.
template <class ARRAY, class SHAPE>
inline
enable_if_t<ArrayMathConcept<ARRAY>::value, ARRAY &&>
forwardPointerND(ARRAY && src, SHAPE const &)
{
    return std::forward<ARRAY>(src);
}

template <int N, class T, class SHAPE>
inline
array_math::ArrayMathExpression<PointerND<N, T>>
forwardPointerND(PointerND<N, T> const & p, SHAPE const & s)
{
    return array_math::ArrayMathExpression<PointerND<N, T>>(p, s);
}

} // namespace array_detail

/********************************************************/
/*                                                      */
/*            consecutivePointerNDFunction()            */
/*                                                      */
/********************************************************/

    // consecutivePointerNDFunction() invokes an optimized algorithm for PointerND with
    // consecutive memory. It returns 'false' if the optimization was impossible (the default),
    // so that the caller can fall-back to a different implementation.
    //
    // This optimization is probably overkill for now, but serves as a proof-of-concept for
    // future optimizations, e.g. via AVX.
template <class P, class SHAPE, class FCT>
constexpr bool consecutivePointerNDFunction(P &&, SHAPE const &, FCT &&, int)
{
    return false;
}

template <class P1, class P2, class SHAPE, class FCT>
constexpr bool consecutivePointerNDFunction(P1 &&, P2 &&, SHAPE const &, FCT &&, int)
{
    return false;
}

template <int N, class T, class SHAPE, class FCT>
inline bool
consecutivePointerNDFunction(const PointerND<N, T> & pn, SHAPE const & shape, FCT &&f, int dim)
{
  return consecutivePointerNDFunction(PointerND<N, T>(pn),shape,std::forward<FCT>(f),dim);
}

template <int N, class T, class SHAPE, class FCT>
inline bool
consecutivePointerNDFunction(PointerND<N, T> && pn, SHAPE const & shape, FCT &&f, int dim)
{
    static_assert(N != 0,
        "consecutivePointerNDFunction(): internal error: N==0 should never happen.");

    auto count = array_detail::isCConsecutive(pn, shape, dim);
    if(count == 0)
        return false;

    auto p = pn.ptr();
    for(ArrayIndex k=0; k<count; ++k, ++p)
        f(*p);
    return true;
}

template <int M, class T, int N, class U, class SHAPE, class FCT>
inline bool
consecutivePointerNDFunction(const PointerND<M, T> & pn1, const PointerND<N, U> & pn2,
                             SHAPE const & shape, FCT &&f, int dim)
{
  return consecutivePointerNDFunction(PointerND<M, T>(pn1),PointerND<N, U>(pn1),shape,std::forward<FCT>(f),dim);
}

template <int M, class T, int N, class U, class SHAPE, class FCT>
inline bool
consecutivePointerNDFunction(PointerND<M, T> && pn1, PointerND<N, U> && pn2,
                             SHAPE const & shape, FCT &&f, int dim)
{
    auto count = array_detail::isCConsecutive(pn1, shape, dim);
    if(count == 0 || array_detail::isCConsecutive(pn2, shape, dim) != count)
        return false;

    auto p1 = pn1.ptr();
    auto p2 = pn2.ptr();
    for(ArrayIndex k=0; k<count; ++k, ++p1, ++p2)
        f(*p1, *p2);
    return true;
}

template <class T, int N, class U, class SHAPE, class FCT>
inline bool
consecutivePointerNDFunction(const PointerND<0, T> & pn1, const PointerND<N, U> & pn2,
                             SHAPE const & shape, FCT &&f, int dim)
{
  return consecutivePointerNDFunction(pn1,PointerND<N, U>(pn2),shape,std::forward<FCT>(f),dim);
}

template <class T, int N, class U, class SHAPE, class FCT>
inline bool
consecutivePointerNDFunction(PointerND<0, T> pn1, PointerND<N, U> && pn2,
                             SHAPE const & shape, FCT &&f, int dim)
{
    auto count = array_detail::isCConsecutive(pn2, shape, dim);
    if(count == 0)
        return false;

    auto p1 = pn1.ptr();
    auto p2 = pn2.ptr();
    for(ArrayIndex k=0; k<count; ++k, ++p2)
        f(*p1, *p2);
    return true;
}

template <int N, class T, class U, class SHAPE, class FCT>
inline bool
consecutivePointerNDFunction(const PointerND<N, T> & pn1, const PointerND<0, U>  & pn2,
                             SHAPE const & shape, FCT &&f, int dim)
{
  return consecutivePointerNDFunction(PointerND<N, T>(pn1),pn2,shape,std::forward<FCT>(f),dim);
}

template <int N, class T, class U, class SHAPE, class FCT>
inline bool
consecutivePointerNDFunction(PointerND<N, T> && pn1, PointerND<0, U> pn2,
                             SHAPE const & shape, FCT &&f, int dim)
{
    auto count = array_detail::isCConsecutive(pn1, shape, dim);
    if(count == 0)
        return false;

    auto p1 = pn1.ptr();
    auto p2 = pn2.ptr();
    for(ArrayIndex k=0; k<count; ++k, ++p1)
        f(*p1, *p2);
    return true;
}

/********************************************************/
/*                                                      */
/*             universalPointerNDFunction()             */
/*                                                      */
/********************************************************/

    // Iterate over a single PointerND in C-order, i.e. with the first dimension
    // in the outer loop and the last dimension in the inner loop. Execute function
    // `f` for the current element in every iteration. The `dim` parameter denotes the
    // dimension of the current loop level, so that the function can be executed
    // recursively.
template <class POINTER_ND, class SHAPE, class FCT,
          VIGRA_REQUIRE<PointerNDConcept<POINTER_ND>::value> >
void
universalPointerNDFunction(POINTER_ND && h, SHAPE const & shape, FCT &&f, int dim = 0)
{
    vigra_assert(dim < shape.size(),
        "universalPointerNDFunction(): internal error: dim >= shape.size() should never happen.");

    if(consecutivePointerNDFunction(std::forward<POINTER_ND>(h), shape, std::forward<FCT>(f), dim))
        return;

    auto N = shape[dim];
    if(dim == shape.size() - 1)
    {
        for(ArrayIndex k=0; k<N; ++k, h.inc(dim))
            f(*h);
    }
    else
    {
        for(ArrayIndex k=0; k<N; ++k, h.inc(dim))
            universalPointerNDFunction(std::forward<POINTER_ND>(h), shape, std::forward<FCT>(f), dim+1);
    }
    h.move(dim, -N);
}

    // Iterate over two PointerND instances in C-order (first dimension in outer loop,
    // last dimension in inner loop) and call binary function `f` in every iteration.
template <class POINTER_ND1, class POINTER_ND2, class SHAPE, class FCT,
          VIGRA_REQUIRE<PointerNDConcept<POINTER_ND1>::value && PointerNDConcept<POINTER_ND2>::value> >
void
universalPointerNDFunction(POINTER_ND1 && h1, POINTER_ND2 && h2, SHAPE const & shape,
                           FCT &&f, int dim = 0)
{
    vigra_assert(dim < shape.size(),
        "universalPointerNDFunction(): internal error: dim >= shape.size() should never happen.");

    if(consecutivePointerNDFunction(std::forward<POINTER_ND1>(h1), std::forward<POINTER_ND2>(h2), shape, std::forward<FCT>(f), dim))
        return;

    auto N = shape[dim];
    if(dim == shape.size() - 1)
    {
        for(ArrayIndex k=0; k<N; ++k, h1.inc(dim), h2.inc(dim))
            f(*h1, *h2);
    }
    else
    {
        for(ArrayIndex k=0; k<N; ++k, h1.inc(dim), h2.inc(dim))
            universalPointerNDFunction(std::forward<POINTER_ND1>(h1), std::forward<POINTER_ND2>(h2), shape, std::forward<FCT>(f), dim+1);
    }
    h1.move(dim, -N);
    h2.move(dim, -N);
}

/********************************************************/
/*                                                      */
/*              reversePointerNDFunction()              */
/*                                                      */
/********************************************************/

    // Iterate over two PointerND instances in reverse C-order (first dimension in outer loop,
    // last dimension in inner loop) and call binary function `f` in every iteration.
template <class POINTER_ND1, class POINTER_ND2, class SHAPE, class FCT,
          VIGRA_REQUIRE<PointerNDConcept<POINTER_ND1>::value && PointerNDConcept<POINTER_ND2>::value> >
void
reversePointerNDFunction(POINTER_ND1 && h1, POINTER_ND2 && h2, SHAPE const & shape,
                         FCT &&f, int dim = 0)
{
    vigra_assert(dim < shape.size(),
        "reversePointerNDFunction(): internal error: dim >= shape.size() should never happen.");

    auto N = shape[dim];
    h1.move(dim, N-1);
    h2.move(dim, N-1);
    if(dim == shape.size() - 1)
    {

        for(ArrayIndex k=N; k>0; --k, h1.dec(dim), h2.dec(dim))
            f(*h1, *h2);
    }
    else
    {
        for(ArrayIndex k=N; k>0; --k, h1.dec(dim), h2.dec(dim))
            reversePointerNDFunction(std::forward<POINTER_ND1>(h1), std::forward<POINTER_ND2>(h2), shape, std::forward<FCT>(f), dim+1);
    }
    h1.inc(dim);
    h2.inc(dim);
}

/********************************************************/
/*                                                      */
/*               universalArrayNDFunction()             */
/*                                                      */
/********************************************************/

    // Apply a lambda function to `src` and `target`, possibly storing the reault in
    // `target`. The function optimizes loop order to maximize cache locality and
    // takes care of overlapping memory and singleton axes.
template <class TARGET, class ARRAY_LIKE, class FCT>
enable_if_t<ArrayNDConcept<TARGET>::value && ArrayLikeConcept<ARRAY_LIKE>::value>
universalArrayNDFunction(TARGET && target, ARRAY_LIKE && src, FCT &&f,
                         std::string func_name)
{
    using namespace array_detail;

    typedef typename std::remove_reference<TARGET>::type      ARRAY1;
    typedef typename std::remove_reference<ARRAY_LIKE>::type  ARRAY2;

    static const int dimension = array_math::ArrayMathUnifyDimension<ARRAY1, ARRAY2>::value;

    // find the common shape, possibly expanding singleton axes
    Shape<dimension> shape = target.shape();
    vigra_precondition(unifyShape(shape, src), func_name + ": shape mismatch.");

    Shape<dimension> p(tags::size = shape.size());
    if (shape.size() > 1)
    {
        // optimize loop order
        Shape<dimension> strides(tags::size = shape.size());
        // determine principal strides so that the optmization also works when
        // the arrays have singleton axes
        principalStrides(strides, target, src);
        p = detail::permutationToOrder(strides, C_ORDER);
    }

    auto tp = target.pointer_nd(p);
    auto sp = src.pointer_nd(p);
    shape   = shape.transpose(p);

    // Take care of overlapping arrays unless the target is read-only
    // (source data could otherwise be overwritten before reading).
    typedef typename std::remove_reference<decltype(*tp)>::type TARGET_VALUE;
    static const bool read_only_target = std::is_const<TARGET_VALUE>::value ||
                                         !std::is_reference<decltype(*tp)>::value;
    MemoryOverlap overlap = read_only_target
                                ? NoMemoryOverlap
                                : checkMemoryOverlap(target.memoryRange(), src);
    if (overlap == NoMemoryOverlap)
    {
        universalPointerNDFunction(tp, sp, shape, std::forward<FCT>(f));
    }
    else if (sp.compatibleStrides(tp.byte_strides()))
    {
        if (overlap & TargetOverlapsLeft) // target below source => work forward
            universalPointerNDFunction(tp, sp, shape, std::forward<FCT>(f));
        else                              // target above source => work backward
            reversePointerNDFunction(tp, sp, shape, std::forward<FCT>(f));
    }
    else
    {
        // hopeless overlap, create a temporary copy of src
        ArrayND<ARRAY2::dimension, typename ARRAY2::value_type> tmp(forwardPointerND(sp, shape));
        universalPointerNDFunction(tp, tmp.pointer_nd(), shape, std::forward<FCT>(f));
    }
}

    // Apply a lambda function to `target`, which may be an array or
    // expression template. The function optimizes loop order to maximize
    // cache locality and takes care of singleton axes.
template <class TARGET, class FCT>
enable_if_t<ArrayLikeConcept<TARGET>::value>
universalArrayNDFunction(TARGET && target, FCT && f,
                         std::string func_name)
{
    typedef typename std::remove_reference<TARGET>::type ARRAY;

    // find the common shape, possibly expanding singleton axes
    Shape<ARRAY::dimension> shape(tags::size = target.ndim());
    vigra_precondition(target.unifyShape(shape), func_name + ": shape mismatch.");

    if (shape.size() > 1)
    {
        // optimize loop order
        Shape<ARRAY::dimension> strides(tags::size = shape.size());
        // determine principal strides so that the optmization also works when
        // the arrays have singleton axes
        target.principalStrides(strides);
        auto p = detail::permutationToOrder(strides, C_ORDER);
        universalPointerNDFunction(target.pointer_nd(p), shape.transpose(p), std::forward<FCT>(f));
    }
    else
    {
        universalPointerNDFunction(target.pointer_nd(), shape, std::forward<FCT>(f));
    }
}

} // namespace vigra

#endif // VIGRA2_POINTER_ND_HXX
