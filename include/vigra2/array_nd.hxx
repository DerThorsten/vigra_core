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

#ifndef VIGRA2_ARRAY_ND_HXX
#define VIGRA2_ARRAY_ND_HXX

#include <vector>
#include <utility>
#include "config.hxx"
#include "concepts.hxx"
#include "tinyarray.hxx"
#include "shape.hxx"
#include "box.hxx"
#include "handle_nd.hxx"
#include "iterator_nd.hxx"
#include "array_math.hxx"
#include "axistags.hxx"

// Bounds checking Macro used if VIGRA_CHECK_BOUNDS is defined.
#ifdef VIGRA_CHECK_BOUNDS
#define VIGRA_ASSERT_INSIDE(diff) \
  vigra_precondition(this->isInside(diff), "Index out of bounds")
#else
#define VIGRA_ASSERT_INSIDE(diff)
#endif

namespace vigra {

using std::swap;

template <int N, class T>
class ArrayViewND;

template <int N, class T, class Alloc = std::allocator<T> >
class ArrayND;

template <class T>
class FFTWComplex;

template <class T, unsigned int R, unsigned int G, unsigned int B>
class RGBValue;

namespace array_detail {

struct UnsuitableTypeForExpandElements {};

template <class T>
struct VectorElementSize;

template <class T>
struct VectorElementSize<std::complex<T> >
{
    static int size(std::complex<T> const *)
    {
        return 2;
    }
};

template <class T>
struct VectorElementSize<FFTWComplex<T> >
{
    static int size(FFTWComplex<T> const *)
    {
        return 2;
    }
};

template <class T, int SIZE>
struct VectorElementSize<TinyArray<T, SIZE> >
{
    static int size(TinyArray<T, SIZE> const * t)
    {
        return t ? t->size() : 1;
    }
};

template <class T, unsigned int R, unsigned int G, unsigned int B>
struct VectorElementSize<RGBValue<T, R, G, B> >
{
    static int size(RGBValue<T, R, G, B> const *)
    {
        return 3;
    }
};

#define VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(TYPE) \
template <>  \
struct VectorElementSize<TYPE> \
{ \
    static int size(TYPE const *) \
    { \
        return 1; \
    } \
};

VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(bool)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(char)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(signed char)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(signed short)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(signed int)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(signed long)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(signed long long)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(unsigned char)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(unsigned short)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(unsigned int)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(unsigned long)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(unsigned long long)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(float)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(double)
VIGRA_DEFINE_VECTOR_ELEMENT_SIZE(long double)

#undef VIGRA_DEFINE_VECTOR_ELEMENT_SIZE

template <int M>
inline ArrayIndex
scanOrderToOffset(ArrayIndex d,
                  Shape<M> const & shape,
                  Shape<M> const & strides)
{
    ArrayIndex res = 0;
    for(int k=0; k<shape.size(); ++k)
    {
        res += strides[k] * (d % shape[k]);
        d /= shape[k];
    }
    return res;
}

} // namespace array_detail

template <int N, class T>
class ArrayViewND
: public ArrayNDTag
{
  protected:
    enum Flags { ConsecutiveMemory = 1,
                 OwnsMemory = 2
               };

  public:
        /** the array's nominal dimensionality N.
         */
    static const int dimension = N;

        /** the array's actual dimensionality.
            This ensures that ArrayViewND can also be used for
            scalars (that is, when <tt>N == 0</tt>). Calculated as:<br>
            \code
            actual_dimension = (N==0) ? 1 : N
            \endcode
         */
    static const int actual_dimension = (N==0) ? 1 : N;

        /** the array's value type
         */
    typedef T value_type;

        /** the read-only variants of the array's value type
         */
    typedef typename std::add_const<value_type>::type  const_value_type;

        /** reference type (result of operator[])
         */
    typedef value_type &reference;

        /** const reference type (result of operator[] const)
         */
    typedef const_value_type &const_reference;

        /** pointer type
         */
    typedef value_type *pointer;

        /** const pointer type
         */
    typedef const_value_type *const_pointer;

        /** difference type (used for multi-dimensional offsets and indices)
         */
    typedef Shape<actual_dimension> difference_type;

        /** key type (argument of index operator array[i] -- same as difference_type)
         */
    typedef difference_type key_type;

        /** size type
         */
    typedef difference_type size_type;

        /** difference and index type for a single dimension
         */
    typedef ArrayIndex difference_type_1;

        /** type for axistag of every dimension
         */
    typedef TinyArray<AxisTag, actual_dimension> axistags_type;

        /** the array view's own type
         */
    typedef ArrayViewND view_type;

        /** the array's handle type
         */
    typedef HandleND<N, value_type> handle_type;

         /** scan-order iterator (ArrayNDIterator) type
         */
    typedef ArrayNDIterator<actual_dimension, T> iterator;

        /** const scan-order iterator (ArrayNDIterator) type
         */
    typedef ArrayNDIterator<actual_dimension, T const> const_iterator;

        // /** the matrix type associated with this array.
         // */
    // typedef ArrayND <N, T> matrix_type;

  protected:

    typedef typename difference_type::value_type diff_zero_t;

        /** the shape of the array pointed to.
        */
    difference_type shape_;

        /** the strides (offset between consecutive elements) for every dimension.
        */
    difference_type strides_;

        /** the axistags for every dimension.
        */
    axistags_type axistags_;

        /** pointer to the array.
         */
    pointer data_;

        /** keep track of various properties
         */
    unsigned flags_;

    void assignImpl(ArrayViewND const & rhs)
    {
        if(data_ == 0)
        {
            shape_     = rhs.shape();
            strides_   = rhs.strides();
            axistags_  = rhs.axistags();
            data_      = const_cast<pointer>(rhs.data());
            flags_     = rhs.flags() & ~OwnsMemory;
        }
        else
        {
            copyImpl(rhs);
        }
    }

    template <int M, class U>
    void copyImpl(ArrayViewND<M, U> const & rhs)
    {
        vigra_precondition(shape() == rhs.shape(),
            "ArrayViewND::operator=(ArrayViewND const &): shape mismatch.");
        array_detail::genericArrayFunction(*this, rhs,
            [](value_type & v, U const & u)
            {
                v = detail::RequiresExplicitCast<value_type>::cast(u);
            });
    }

        // ensure that singleton axes have unit stride
    void fixSingletonStrides()
    {
        for(int k=0; k<ndim(); ++k)
            if(shape_[k] == 1)
                strides_[k] = 1;
    }

    unsigned isConsecutiveImpl() const
    {
        return (&operator[](shape_ - 1) == &data_[size()-1])
                     ? ConsecutiveMemory
                     : 0;
    }

    void swapImpl(ArrayViewND & rhs)
    {
        if(this != &rhs)
        {
            shape_.swap(rhs.shape_);
            strides_.swap(rhs.strides_);
            axistags_.swap(rhs.axistags_);
            vigra::swap(data_    , rhs.data_);
            vigra::swap(flags_   , rhs.flags_);
        }
    }

public:

        /** default constructor: create an invalid view,
         * i.e. hasData() returns false and size() is zero.
         */
    ArrayViewND()
    : shape_()
    , strides_()
    , axistags_()
    , data_(0)
    , flags_(0)
    {}

    ArrayViewND(ArrayViewND const & other)
    : shape_(other.shape())
    , strides_(other.strides())
    , axistags_(other.axistags())
    , data_(const_cast<pointer>(other.data()))
    , flags_(other.flags() & ~OwnsMemory)
    {}

    template <int M>
    ArrayViewND(ArrayViewND<M, T> const & other)
    : shape_(other.shape())
    , strides_(other.strides())
    , axistags_(other.axistags())
    , data_(const_cast<pointer>(other.data()))
    , flags_(other.flags() & ~OwnsMemory)
    {
        static_assert(CompatibleDimensions<M, N>::value,
            "ArrayViewND<N>(ArrayViewND<M>): ndim mismatch.");
    }

    // FIXME: add constructor from explicit channel to vector pixels

        /** construct from shape and pointer
         */
    ArrayViewND(const difference_type &shape,
                const_pointer ptr,
                MemoryOrder order = C_ORDER)
    : ArrayViewND(shape, shapeToStrides(shape, order), ptr)
    {}

        /** construct from shape, axistags, and pointer
         */
    ArrayViewND(const difference_type &shape,
                const axistags_type   &axistags,
                const_pointer ptr,
                MemoryOrder order = C_ORDER)
    : ArrayViewND(shape, shapeToStrides(shape, order), axistags, ptr)
    {}

        /** Construct from shape, strides (offset of a sample to the
            next) for every dimension, and pointer.  (Note that
            strides are not given in bytes, but in offset steps of the
            respective pointer type.)
         */
    ArrayViewND(const difference_type &shape,
                const difference_type &strides,
                const_pointer ptr)
    : ArrayViewND(shape, strides,
                  axistags_type(tags::size = shape.size(), tags::axis_unknown), ptr)
    {}

        /** Construct from shape, strides (offset of a sample to the
            next), axistags for every dimension, and pointer.  (Note that
            strides are not given in bytes, but in offset steps of the
            respective pointer type.)
         */
    ArrayViewND(const difference_type &shape,
                const difference_type &strides,
                const axistags_type   &axistags,
                const_pointer ptr)
    : shape_(shape)
    , strides_(strides)
    , axistags_(axistags)
    , data_(const_cast<pointer>(ptr))
    , flags_(isConsecutiveImpl())
    {
        fixSingletonStrides();
    }

        /* Construct 0-dimensional array from 0-dimensional shape/stride
           (needed in functions recursing on ndim()).
         */
    ArrayViewND(const Shape<0> &,
                const Shape<0> &,
                const TinyArray<AxisTag, 0> &,
                const_pointer ptr)
    : shape_{1}
    , strides_{1}
    , axistags_{tags::axis_unknown}
    , data_(const_cast<pointer>(ptr))
    , flags_(ConsecutiveMemory)
    {
        static_assert(N <= 0,
            "ArrayViewND(): 0-dimensional constructor can only be called when N == 1.");
    }

        /** Assignment. There are 3 cases:

            <ul>
            <li> When this <tt>ArrayViewND</tt> does not point to valid data
                 (e.g. after default construction), it becomes a new view of \a rhs.
            <li> Otherwise, when the shapes of the two arrays match, the contents
                 (i.e. the elements) of \a rhs are copied.
            <li> Otherwise, a <tt>PreconditionViolation</tt> exception is thrown.
            </ul>
         */
    ArrayViewND & operator=(ArrayViewND const & rhs)
    {
        if(this != &rhs)
            assignImpl(rhs);
        return *this;
    }

        /** Assignment of a scalar.
         */
    ArrayViewND &
    operator=(value_type const & u)
    {
        array_detail::genericArrayFunction(*this,
                                           [u](value_type & v) { v = u; });
        return *this;
    }

#ifdef DOXYGEN
        /** Assignment of a differently typed array or an array expression. It copies the elements
            of\a rhs or fails with a <tt>PreconditionViolation</tt> exception when
            the shapes do not match.
         */
    template<class ARRAY>
    ArrayViewND & operator=(ARRAY const & rhs);

        /** Add-assignment of a differently typed array or an array expression.
            It adds the elements of \a rhs or fails with a <tt>PreconditionViolation</tt>
            exception when the shapes do not match.
         */
    template <class ARRAY>
    ArrayViewND & operator+=(ARRAY const & rhs);

        /** Subtract-assignment of a differently typed array or an array expression.
            It subtracts the elements of \a rhs or fails with a <tt>PreconditionViolation</tt>
            exception when the shapes do not match.
         */
    template <class ARRAY>
    ArrayViewND & operator-=(ARRAY const & rhs);

        /** Multiply-assignment of a differently typed array or an array expression.
            It multiplies with the elements of \a rhs or fails with a
            <tt>PreconditionViolation</tt> exception when the shapes do not match.
         */
    template <class ARRAY>
    ArrayViewND & operator*=(ARRAY const & rhs);

        /** Divide-assignment of a differently typed array or an array expression.
            It divides by the elements of \a rhs or fails with a <tt>PreconditionViolation</tt>
            exception when the shapes do not match.
         */
    template <class ARRAY>
    ArrayViewND & operator/=(ARRAY const & rhs);

#else

#define VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(OP) \
    template <class ARG> \
    enable_if_t<ArrayNDConcept<ARG>::value || ArrayMathConcept<ARG>::value, \
                ArrayViewND &> \
    operator OP(ARG const & rhs) \
    { \
        typedef typename typename ARG::value_type U; \
        static_assert(std::is_convertible<U, value_type>::value, \
            "ArrayViewND::operator" #OP "(ARRAY const &): value_types of lhs and rhs are incompatible."); \
            \
        vigra_precondition(shape() == rhs.shape(), \
            "ArrayViewND::operator" #OP "(ArrayViewND const &): shape mismatch."); \
        array_detail::genericArrayFunction(*this, rhs, \
            [](value_type & v, U const & u) \
            { \
                v OP detail::RequiresExplicitCast<value_type>::cast(u); \
            }); \
        return *this; \
    }

    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(=)
    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(+=)
    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(-=)
    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(*=)
    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(/=)

#undef VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT

#endif // DOXYGEN

        /** Add-assignment of a scalar.
         */
    ArrayViewND & operator+=(value_type const & u)
    {
        array_detail::genericArrayFunction(*this,
                                           [u](value_type & v) { v += u; });
        return *this;
    }

        /** Subtract-assignment of a scalar.
         */
    ArrayViewND & operator-=(value_type const & u)
    {
        array_detail::genericArrayFunction(*this,
                                           [u](value_type & v) { v -= u; });
        return *this;
    }

        /** Multiply-assignment of a scalar.
         */
    ArrayViewND & operator*=(value_type const & u)
    {
        array_detail::genericArrayFunction(*this,
                                           [u](value_type & v) { v *= u; });
        return *this;
    }

        /** Divide-assignment of a scalar.
         */
    ArrayViewND & operator/=(value_type const & u)
    {
        array_detail::genericArrayFunction(*this,
                                           [u](value_type & v) { v /= u; });
        return *this;
    }

        /** Access element.
         */
    reference operator[](const difference_type &d)
    {
        VIGRA_ASSERT_INSIDE(d);
        return data_ [dot (d, strides_)];
    }

        /** Access element via scalar index. Only allowed if
            <tt>isConsecutive() == true</tt> or <tt>ndim() <= 1</tt>.
         */
    reference operator[](const difference_type_1 &d)
    {
        if(isConsecutive())
            return data_[d];
        if(ndim() <= 1)
            return data_[d*strides_[0]];
        vigra_precondition(false,
            "ArrayViewND::operator[](int) forbidden for strided multi-dimensional arrays.");
    }

        /** Get element.
         */
    const_reference operator[](const difference_type &d) const
    {
        VIGRA_ASSERT_INSIDE(d);
        return data_ [dot (d, strides_)];
    }

        /** Get element via scalar index. Only allowed if
            <tt>isConsecutive() == true</tt> or <tt>ndim() <= 1</tt>.
         */
    const_reference operator[](const difference_type_1 &d) const
    {
        if(isConsecutive())
            return data_[d];
        if(ndim() <= 1)
            return data_[d*strides_[0]];
        vigra_precondition(false,
            "ArrayViewND::operator[](int) forbidden for strided multi-dimensional arrays.");
    }

        /** 1D array access. Use only if <tt>ndim() <= 1</tt>.
         */
    reference operator()(difference_type_1 i)
    {
        vigra_assert(ndim() <= 1,
                      "ArrayViewND::operator()(int): only allowed if ndim() <= 1");
        return data_[i*strides_[0]];
    }

        /** N-D array access. Number of indices must match <tt>ndim()</tt>.
         */
    template <class ... INDICES>
    reference operator()(difference_type_1 i0, difference_type_1 i1,
                         INDICES ... i)
    {
        static const int M = 2 + sizeof...(INDICES);
        vigra_assert(ndim() == M,
            "ArrayViewND::operator()(INDICES): number of indices must match ndim().");
        return data_[dot(Shape<M>(i0, i1, i...), strides_)];
    }

        /** 1D array access. Use only if <tt>ndim() <= 1</tt>.
         */
    const_reference operator()(difference_type_1 i) const
    {
        vigra_assert(ndim() <= 1,
                      "ArrayViewND::operator()(int): only allowed if ndim() <= 1");
        return data_[i*strides_[0]];
    }

        /** N-D array access. Number of indices must match <tt>ndim()</tt>.
         */
    template <class ... INDICES>
    const_reference operator()(difference_type_1 i0, difference_type_1 i1,
                               INDICES ... i) const
    {
        static const int M = 2 + sizeof...(INDICES);
        vigra_assert(ndim() == M,
            "ArrayViewND::operator()(INDICES): number of indices must match ndim().");
        return data_[dot(Shape<M>(i0, i1, i...), strides_)];
    }

        /** Bind 'axis' to 'index'.

            This reduces the dimensionality of the array by one.

            <b>Usage:</b>
            \code
            // create a 3D array of size 40x30x20
            ArrayND<3, double> array3({40, 30, 20});

            // get a 2D array by fixing index 2 to 15
            ArrayViewND<2, double> array2 = array3.bind(2, 15);
            \endcode
         */
    ArrayViewND<((N < 0) ? runtime_size : N-1), T>
    bind(difference_type_1 axis, difference_type_1 index) const
    {
        typedef ArrayViewND<((N < 0) ? runtime_size : N-1), T> Result;

        vigra_assert(0 <= axis && axis < ndim() && 0 <= index && index < shape_[axis],
            "ArrayViewND::bind(): index out of range.");

        difference_type point(tags::size = ndim(), 0);
        point[axis] = index;
        if (ndim() == 1)
        {
            Shape<Result::actual_dimension> shape{ 1 }, stride{ 1 };
            TinyArray<AxisTag, Result::actual_dimension> axistags{ tags::axis_unknown };
            return Result(shape, stride, axistags, &operator[](point));
        }
        else
        {
            return Result(shape_.erase(axis),
                          strides_.erase(axis),
                          axistags_.erase(axis),
                          &operator[](point));
        }
    }

        /** Bind the dimensions 'axes' to 'indices.

            Only applicable when <tt>M <= N</tt> or <tt>N == runtime_size</tt>.
            The elements of 'axes' must be unique, contained in the interval
            <tt>0 <= element < ndim()<//t> and be sorted in ascending order.
            The elements of 'indices' must be in the valid range of the
            corresponding axes.
         */
    template <int M>
    ArrayViewND<((N < 0) ? runtime_size : N-M), T>
    bind(Shape<M> const & axes, Shape<M> const & indices) const
    {
        static_assert(N == runtime_size || M <= N,
            "ArrayViewND::bind(Shape<M>): M <= N required.");
        return bind(axes.back(), indices.back())
                  .bind(axes.pop_back(), indices.pop_back());
    }


    ArrayViewND<((N < 0) ? runtime_size : N-1), T>
    bind(Shape<1> const & a, Shape<1> const & i) const
    {
        return bind(a[0], i[0]);
    }

    ArrayViewND const &
    bind(Shape<0> const &, Shape<0> const &) const
    {
        return *this;
    }

    ArrayViewND<runtime_size, T>
    bind(Shape<runtime_size> const & axes, Shape<runtime_size> const & indices) const
    {
        vigra_precondition(axes.size() == indices.size(),
            "ArrayViewND::bind(): size mismatch betwwen 'axes' and 'indices'.");
        vigra_precondition(axes.size() <= ndim(),
            "ArrayViewND::bind(): axes.size() <= ndim() required.");

        ArrayViewND<runtime_size, T> a(*this);
        if(axes.size() == 0)
            return a;
        else
            return a.bind(axes.back(), indices.back())
                       .bind(axes.pop_back(), indices.pop_back());
    }

        /** Bind the channel axis to index d.
            This calls <tt>array.bind(array.channelAxis(), d)</tt>
            when a channel axis is defined and throws an error otherwise.
            \endcode
         */
    ArrayViewND<((N < 0) ? runtime_size : N-1), T>
    bindChannel(difference_type_1 d) const
    {
        int m = channelAxis();

        vigra_assert(m != tags::no_channel_axis,
            "ArrayViewND::bindChannel(): array has no channel axis.");

        return bind(m, d);
    }

        /** Bind the first 'indices.size()' dimensions to 'indices'.

            Only applicable when <tt>M <= N</tt> or <tt>N == runtime_size</tt>.
         */
    template <int M>
    auto
    bindLeft(Shape<M> const & indices) const -> decltype(bind(indices, indices))
    {
        return bind(Shape<M>::range(indices.size()), indices);
    }

        /** Bind the last 'indices.size()' dimensions to 'indices'.

            Only applicable when <tt>M <= N</tt> or <tt>N == runtime_size</tt>.
         */
    template <int M>
    auto
    bindRight(Shape<M> const & indices) const -> decltype(bind(indices, indices))
    {
        return bind(Shape<M>::range(indices.size()) + ndim() - indices.size(),
                    indices);
    }

        /** Create a view to channel 'i' of a vector-like value type. Possible value types
            (of the original array) are: \ref TinyVector, \ref RGBValue, \ref FFTWComplex,
            and <tt>std::complex</tt>. The function can be applied whenever the array's
            element type <tt>T</tt> defines an embedded type <tt>T::value_type</tt> which
            becomes the return type of <tt>bindElementChannel()</tt>.


            <b>Usage:</b>
            \code
                ArrayND<2, RGBValue<float> > rgb_image({h,w});

                ArrayViewND<2, float> red   = rgb_image.bindElementChannel(0);
                ArrayViewND<2, float> green = rgb_image.bindElementChannel(1);
                ArrayViewND<2, float> blue  = rgb_image.bindElementChannel(2);
            \endcode
        */
    template <class U=T,
              VIGRA_REQUIRE<!std::is_scalar<U>::value> >
    ArrayViewND<N, typename U::value_type>
    bindElementChannel(difference_type_1 i) const
    {
        vigra_precondition(0 <= i &&
                           i < array_detail::VectorElementSize<T>::size(data_),
            "ArrayViewND::bindElementChannel(i): 'i' out of range.");
        return expandElements(0).bind(0, i);
    }

        /** Create a view where a vector-like element type is expanded into a new
            array dimension. The new dimension is inserted at index position 'd',
            which must be between 0 and N inclusive.

            Possible value types of the original array are: \ref TinyArray, \ref RGBValue,
            \ref FFTWComplex, <tt>std::complex</tt>, and the built-in number types (in this
            case, <tt>expandElements</tt> is equivalent to <tt>insertSingletonDimension</tt>).
            The function requires the array's element type <tt>T</tt> to define
            an embedded type <tt>T::value_type</tt>.

            <b>Usage:</b>
            \code
                ArrayND<2, TinyArray<float, 3> > rgb_image({h, w});

                ArrayViewND<3, float> multiband_image = rgb_image.expandElements(2);
            \endcode
        */
    template <class U=T,
              VIGRA_REQUIRE<!std::is_scalar<U>::value> >
    ArrayViewND<(N == runtime_size ? runtime_size : N+1), typename U::value_type>
    expandElements(difference_type_1 d) const
    {
        using Value  = typename T::value_type;
        using Result = ArrayViewND <(N == runtime_size ? runtime_size : N + 1), Value>;

        vigra_precondition(0 <= d && d <= ndim(),
            "ArrayViewND::expandElements(d): 0 <= 'd' <= ndim() required.");

        int s = array_detail::VectorElementSize<T>::size(data_);
        return Result(shape_.insert(d, s),
                      (strides_ * s).insert(d, 1),
                      axistags_.insert(d, tags::axis_c),
                      reinterpret_cast<Value*>(data_));
    }

        /** Create a view with an explicit channel axis at index \a d.

            There are three cases:
            <ul>
            <li> If the array's <tt>value_type</tt> is scalar, and the array already
                 has an axis marked as channel axis, the array is transposed such
                 that the channel axis is at index \a d.

            <li> If the array's <tt>value_type</tt> is scalar, and the array does
                 not have a channel axis, the function
                 <tt>insertSingletonDimension(d, tags::axis_c)</tt> is called.

            <li> If the array's <tt>value_type</tt> is  vectorial, the function
                 <tt>expandElements(d)</tt> is called.
            </ul>
            Thus, the function can be called repeatedly without error.

            <b>Usage:</b>
            \code
                ArrayND<2, TinyArray<float, 3> > rgb_image({h, w});

                ArrayViewND<3, float> multiband_image = rgb_image.ensureChannelAxis(2);
                assert(multiband_image.channelAxis() == 3);
            \endcode
        */
    template <class U=T,
              VIGRA_REQUIRE<!std::is_scalar<U>::value> >
    ArrayViewND<runtime_size, typename U::value_type>
    ensureChannelAxis(difference_type_1 d) const
    {
        return expandElements(d);
    }

    template <class U=T,
              VIGRA_REQUIRE<std::is_scalar<U>::value> >
    ArrayViewND<runtime_size, T>
    ensureChannelAxis(difference_type_1 d) const
    {
        vigra_precondition(d >= 0,
            "ArrayViewND::ensureChannelAxis(d): d >= 0 required.");
        int c = channelAxis();
        if(c == d)
            return *this;
        if(c < 0)
            return insertSingletonDimension(d, tags::axis_c);
        vigra_precondition(d < ndim(),
            "ArrayViewND::ensureChannelAxis(d): d < ndim() required.");
        auto permutation = Shape<>::range(ndim()).erase(c).insert(d, c);
        return transpose(permutation);
    }

        /** Add a singleton dimension (dimension of length 1).

            Singleton dimensions don't change the size of the data, but introduce
            a new index that can only take the value 0. This is mainly useful for
            the 'reduce mode' of transformMultiArray() and combineTwoMultiArrays(),
            because these functions require the source and destination arrays to
            have the same number of dimensions.

            The range of \a i must be <tt>0 <= i <= N</tt>. The new dimension will become
            the i'th index, and the old indices from i upwards will shift one
            place to the right.

            <b>Usage:</b>

            Suppose we want have a 2D array and want to create a 1D array that contains
            the row average of the first array.
            \code
            typedef MultiArrayShape<2>::type Shape2;
            ArrayND<2, double> original(Shape2(40, 30));

            typedef MultiArrayShape<1>::type Shape1;
            ArrayND<1, double> rowAverages(Shape1(30));

            // temporarily add a singleton dimension to the destination array
            transformMultiArray(srcMultiArrayRange(original),
                                destMultiArrayRange(rowAverages.insertSingletonDimension(0)),
                                FindAverage<double>());
            \endcode
         */
    ArrayViewND <(N < 0) ? runtime_size : N+1, T>
    insertSingletonDimension(difference_type_1 i,
                             AxisTag tag = tags::axis_unknown) const
    {
        typedef ArrayViewND <(N < 0) ? runtime_size : N+1, T> Result;
        return Result(shape_.insert(i, 1), strides_.insert(i, 1),
                      axistags_.insert(i, tag), data_);
    }
        // /** create a multiband view for this array.

            // The type <tt>ArrayViewND<N, Multiband<T> ></tt> tells VIGRA
            // algorithms which recognize the <tt>Multiband</tt> modifier to
            // interpret the outermost (last) dimension as a channel dimension.
            // In effect, these algorithms will treat the data as a set of
            // (N-1)-dimensional arrays instead of a single N-dimensional array.
        // */
    // ArrayViewND<N, Multiband<value_type>, StrideTag> multiband() const
    // {
        // return ArrayViewND<N, Multiband<value_type>, StrideTag>(*this);
    // }

        /** Create a view to the diagonal elements of the array.

            This produces a 1D array view whose size equals the size
            of the shortest dimension of the original array.

            <b>Usage:</b>
            \code
            // create a 3D array of size 40x30x20
            ArrayND<3, double> array3(Shape<3>(40, 30, 20));

            // get a view to the diagonal elements
            ArrayViewND<1, double> diagonal = array3.diagonal();
            assert(diagonal.shape(0) == 20);
            \endcode
        */
    ArrayViewND<1, T> diagonal() const
    {
        return ArrayViewND<1, T>(Shape<1>(vigra::min(shape_)),
                                 Shape<1>(vigra::sum(strides_)), data_);
    }

        /** create a rectangular subarray that spans between the
            points p and q, where p is in the subarray, q not.
            If an element of p or q is negative, it is subtracted
            from the correspongng shape.

            <b>Usage:</b>
            \code
            // create a 3D array of size 40x30x20
            ArrayND<3, double> array3(Shape<3>(40, 30, 20));

            // get a subarray set is smaller by one element at all sides
            ArrayViewND<3, double> subarray  = array3.subarray(Shape<3>(1,1,1), Shape<3>(39, 29, 19));

            // specifying the end point with a vector of '-1' is equivalent
            ArrayViewND<3, double> subarray2 = array3.subarray(Shape<3>(1,1,1), Shape<3>(-1, -1, -1));
            \endcode
        */
    ArrayViewND
    subarray(difference_type p, difference_type q) const
    {
        vigra_precondition(p.size() == ndim() && q.size() == ndim(),
            "ArrayViewND::subarray(): size mismatch.");
        for(int k=0; k<ndim(); ++k)
        {
            if(p[k] < 0)
                p[k] += shape_[k];
            if(q[k] < 0)
                q[k] += shape_[k];
        }
        vigra_precondition(isInside(p) && allLessEqual(p, q) && allLessEqual(q, shape_),
            "ArrayViewND::subarray(): invalid subarray limits.");
        const difference_type_1 offset = dot(strides_, p);
        return ArrayViewND(q - p, strides_, axistags_, data_ + offset);
    }

        /** Transpose an array. If N==2, this implements the usual matrix transposition.
            For N > 2, it reverses the order of the indices.

            <b>Usage:</b><br>
            \code
            typedef ArrayND<2, double>::difference_type Shape;
            ArrayND<2, double> array(10, 20);

            ArrayViewND<2, double> transposed = array.transpose();

            for(int i=0; i<array.shape(0), ++i)
                for(int j=0; j<array.shape(1); ++j)
                    assert(array(i, j) == transposed(j, i));
            \endcode
        */
    ArrayViewND<N, T>
    transpose() const
    {
        return ArrayViewND<N, T>(vigra::transpose(shape_),
                                 vigra::transpose(strides_),
                                 vigra::transpose(axistags_), data_);
    }

        /** Permute the dimensions of the array.
            The function exchanges the orer of the array's axes without copying the data.
            Argument\a permutation specifies the desired order such that
            <tt>permutation[k] = j</tt> means that axis <tt>j</tt> in the original array
            becomes axis <tt>k</tt> in the transposed array.

            <b>Usage:</b><br>
            \code
            typedef ArrayND<2, double>::difference_type Shape;
            ArrayND<2, double> array(10, 20);

            ArrayViewND<2, double, StridedArrayTag> transposed = array.transpose(Shape(1,0));

            for(int i=0; i<array.shape(0), ++i)
                for(int j=0; j<array.shape(1); ++j)
                    assert(array(i, j) == transposed(j, i));
            \endcode
        */
    template <int M>
    ArrayViewND
    transpose(Shape<M> const & permutation) const
    {
        static_assert(M == actual_dimension || M == runtime_size || N == runtime_size,
            "ArrayViewND::transpose(): permutation.size() doesn't match ndim().");
        vigra_precondition(permutation.size() == ndim(),
            "ArrayViewND::transpose(): permutation.size() doesn't match ndim().");
        difference_type p(permutation);
        ArrayViewND res(::vigra::transpose(shape_, p),
                        ::vigra::transpose(strides_, p),
                        ::vigra::transpose(axistags_, p),
                        data_);
        return res;
    }

    ArrayViewND
    transpose(MemoryOrder order) const
    {
        return transpose(array_detail::permutationToOrder(shape_, strides_, order));
    }

        /** Check if the array contains only non-zero elements (or if all elements
            are 'true' if the value type is 'bool').
         */
    bool all() const
    {
        bool res = true;
        value_type zero = value_type();
        array_detail::genericArrayFunction(*this,
            [zero, &res](value_type const & v)
            {
                if(v == zero)
                    res = false;
            });
        return res;
    }

        /** Check if the array contains a non-zero element (or an element
            that is 'true' if the value type is 'bool').
         */
    bool any() const
    {
        bool res = false;
        value_type zero = value_type();
        array_detail::genericArrayFunction(*this,
            [zero, &res](value_type const & v)
            {
                if(v != zero)
                    res = true;
            });
        return res;
    }

        /** Find the minimum and maximum element in this array.
            See \ref FeatureAccumulators for a general feature
            extraction framework.
         */
    TinyArray<T, 2> minmax() const
    {
        TinyArray<T, 2> res(NumericTraits<T>::max(), NumericTraits<T>::min());
        array_detail::genericArrayFunction(*this,
            [&res](value_type const & v)
            {
                if(v < res[0])
                    res[0] = v;
                if(res[1] < v)
                    res[1] = v;
            });
        return res;
    }

        // /** Compute the mean and variance of the values in this array.
            // See \ref FeatureAccumulators for a general feature
            // extraction framework.
         // */
    // template <class U>
    // void meanVariance(U * mean, U * variance) const
    // {
        // typedef typename NumericTraits<U>::RealPromote R;
        // R zero = R();
        // triple<double, R, R> res(0.0, zero, zero);
        // detail::reduceOverMultiArray(traverser_begin(), shape(),
                                     // res,
                                     // detail::MeanVarianceReduceFunctor(),
                                     // MetaInt<actual_dimension-1>());
        // *mean     = res.second;
        // *variance = res.third / res.first;
    // }

        /** Compute the sum of the array elements.

            You must provide the type of the result by an explicit template parameter:
            \code
            ArrayND<2, UInt8> A(width, height);

            double sum = A.sum<double>();
            \endcode
         */
    template <class U = PromoteType<T> >
    U sum(U res = U{}) const
    {
        array_detail::genericArrayFunction(*this,
            [&res](value_type const & v)
            {
                res += v;
            });
        return res;
    }

        // /** Compute the sum of the array elements over selected axes.

            // \arg sums must have the same shape as this array, except for the
            // axes along which the sum is to be accumulated. These axes must be
            // singletons. Note that you must include <tt>multi_pointoperators.hxx</tt>
            // for this function to work.

            // <b>Usage:</b>
            // \code
            // #include <vigra/multi_array.hxx>
            // #include <vigra/multi_pointoperators.hxx>

            // ArrayND<2, double> A(Shape2(rows, cols));
            // ... // fill A

            // // make the first axis a singleton to sum over the first index
            // ArrayND<2, double> rowSums(Shape2(1, cols));
            // A.sum(rowSums);

            // // this is equivalent to
            // transformMultiArray(srcMultiArrayRange(A),
                                // destMultiArrayRange(rowSums),
                                // FindSum<double>());
            // \endcode
         // */
    // template <class U, class S>
    // void sum(ArrayViewND<N, U, S> sums) const
    // {
        // transformMultiArray(srcMultiArrayRange(*this),
                            // destMultiArrayRange(sums),
                            // FindSum<U>());
    // }

        /** Compute the product of the array elements.

            You must provide the type of the result by an explicit template parameter:
            \code
            ArrayND<2, UInt8> A(width, height);

            double prod = A.product<double>();
            \endcode
         */
    template <class U = PromoteType<T> >
    U prod(U res = U{1}) const
    {
        array_detail::genericArrayFunction(*this,
            [&res](value_type const & v)
            {
                res *= v;
            });
        return res;
    }

    void swap(ArrayViewND & rhs)
    {
        vigra_precondition(!ownsMemory() && !rhs.ownsMemory(),
            "ArrayViewND::swap(): only allowed when views don't own their memory.");
        swapImpl(rhs);
    }

        /** Swap the data between two ArrayViewND objects.

            The shapes of the two array must match. Both array views
            still point to the same memory as before, just the contents
            are exchanged.
        */
    template <int M, class U>
    void
    swapData(ArrayViewND<M, U> rhs)
    {
        static_assert(M == N || M == runtime_size || N == runtime_size,
            "ArrayViewND::swapData(): incompatible dimensions.");
        vigra_precondition(shape() == rhs.shape(),
            "ArrayViewND::swapData(): shape mismatch.");
        array_detail::genericArrayFunction(*this, rhs,
            [](value_type & v, U & u)
            {
                vigra::swap(u, v);
            });
    }

    template <int M>
    ArrayViewND<M, T>
    reshape(Shape<M> new_shape,
            AxisTags<M> new_axistags = AxisTags<M>{},
            MemoryOrder order = C_ORDER) const
    {
        vigra_precondition(isConsecutive(),
            "ArrayViewND::reshape(): only consecutive arrays can be reshaped.");
        if(M <= 1 && new_shape == Shape<M>{})
        {
            new_shape = Shape<M>{size()};
        }
        vigra_precondition(vigra::prod(new_shape) == size(),
            "ArrayViewND::reshape(): size mismatch between old and new shape.");
        if(new_axistags == AxisTags<M>{})
            new_axistags = AxisTags<M>(tags::size = new_shape.size(), tags::axis_unknown);
        vigra_precondition(M != runtime_size || new_axistags.size() == new_shape.size(),
           "ArrayViewND::reshape(): size mismatch between new shape and axistags.");
        return ArrayViewND<M, T>(new_shape, new_axistags, data_, order);
    }

    template <int M>
    ArrayViewND<M, T>
    reshape(Shape<M> const & new_shape,
            MemoryOrder order) const
    {
        return reshape(new_shape, AxisTags<M>{}, order);
    }

    ArrayViewND<(N == runtime_size ? N : 1), T>
    flatten() const
    {
        return reshape(Shape<(N == runtime_size ? N : 1)>{});
    }

        /** number of the elements in the array.
         */
    difference_type_1 size() const
    {
        return vigra::prod(shape_);
    }

#ifdef DOXYGEN
        /** the array's number of dimensions.
         */
    int ndim() const;
#else
        // Actually, we use some template magic to turn ndim() into a
        // constexpr when it is known at compile time.
    template <int M = N>
    int ndim(enable_if_t<M == runtime_size, bool> = true) const
    {
        return shape_.size();
    }

    template <int M = N>
    constexpr int ndim(enable_if_t<(M > runtime_size), bool> = true) const
    {
        return N;
    }
#endif

        /** the array's shape.
         */
    difference_type const & shape() const
    {
        return shape_;
    }

        /** return the array's shape at a certain dimension.
         */
    difference_type_1 shape(int n) const
    {
        return shape_[n];
    }

        /** return the array's strides for every dimension.
         */
    difference_type const & strides() const
    {
        return strides_;
    }

        /** return the array's stride at a certain dimension.
         */
    difference_type_1 strides(int n) const
    {
        return strides_[n];
    }

        /** return the array's axistags for every dimension.
         */
    axistags_type const & axistags() const
    {
        return axistags_;
    }

        /** return the array's axistag at a certain dimension.
         */
    AxisTag axistags(int n) const
    {
        return axistags_[n];
    }

        /** check whether the given point is in the array range.
         */
    bool isInside(difference_type const & p) const
    {
        return Box<actual_dimension>(shape_).contains(p);
    }

        /** check whether the given point is not in the array range.
         */
    bool isOutside(difference_type const & p) const
    {
        return !isInside(p);
    }

        /** return the pointer to the image data
         */
    pointer data()
    {
        return data_;
    }

        /** return the pointer to the image data
         */
    const_pointer data() const
    {
        return data_;
    }

        /**
         * returns true iff this view refers to valid data,
         * i.e. data() is not a NULL pointer.  (this is false after
         * default construction.)
         */
    bool hasData() const
    {
        return data_ != 0;
    }

    bool isConsecutive() const
    {
        return (flags_ & ConsecutiveMemory) != 0;
    }

    bool ownsMemory() const
    {
        return (flags_ & OwnsMemory) != 0;
    }

    ArrayViewND & setAxistags(axistags_type const & t)
    {
        vigra_assert(t.size() == ndim(),
            "ArrayViewND::setAxistags(): size mismatch.");
        axistags_ = t;
        return *this;
    }

    ArrayViewND & setChannelAxis(int c)
    {
        vigra_assert(0 <= c && c < ndim(),
            "ArrayViewND::setChannelAxis(): index out of range.");
        axistags_[c] = tags::axis_c;
        return *this;
    }

    int channelAxis() const
    {
        for(int k=0; k<ndim(); ++k)
            if(axistags_[k] == tags::axis_c)
                return k;
        return tags::no_channel_axis;
    }

    bool hasChannelAxis() const
    {
        return channelAxis() != tags::no_channel_axis;
    }

    unsigned flags() const
    {
        return flags_;
    }

    handle_type handle() const
    {
        return handle_type(strides_, data_);
    }

    handle_type handle(difference_type const & permutation) const
    {
        return handle_type(vigra::transpose(strides_, permutation), data_);
    }

    handle_type handle(MemoryOrder order) const
    {
        return handle(array_detail::permutationToOrder(shape_, strides_, order));
    }

        /** returns a scan-order iterator pointing
            to the first array element.
        */
    iterator begin(MemoryOrder order)
    {
        return iterator(*this, order);
    }

    iterator begin()
    {
        return iterator(*this,
                  array_detail::permutationToOrder(shape(), strides(), F_ORDER));
    }

        /** returns a const scan-order iterator pointing
            to the first array element.
        */
    const_iterator begin(MemoryOrder order) const
    {
        return const_iterator(*this, order);
    }

    const_iterator begin() const
    {
        return const_iterator(*this,
                  array_detail::permutationToOrder(shape(), strides(), F_ORDER));
    }

        /** returns a scan-order iterator pointing
            beyond the last array element.
        */
    iterator end(MemoryOrder order)
    {
        return begin(order).end();
    }

    iterator end()
    {
        return begin().end();
    }

        /** returns a const scan-order iterator pointing
            beyond the last array element.
        */
    const_iterator end(MemoryOrder order) const
    {
        return begin(order).end();
    }

    const_iterator end() const
    {
        return begin().end();
    }

    template <int M = runtime_size>
    ArrayViewND<M, T> view() const
    {
        vigra_precondition(M == runtime_size || M == ndim(),
            "ArrayViewND::view(): desired dimension is incompatible with ndim().");
        return ArrayViewND<M, T>(Shape<M>(shape_.begin(), shape_.begin()+ndim()),
                                 Shape<M>(strides_.begin(), strides_.begin()+ndim()),
                                 AxisTags<M>(axistags_.begin(), axistags_.begin()+ndim()),
                                 data());
    }
};

template <int N, class T>
SquaredNormType<ArrayViewND<N, T> >
squaredNorm(ArrayViewND<N, T> const & a)
{
    auto res = SquaredNormType<ArrayViewND<N, T> >();
    array_detail::genericArrayFunction(a,
        [&res](T const & v)
        {
            res += v*v;
        });
    return res;
}

    /** Compute various norms of the given array.
        The norm is determined by parameter \a type:

        <ul>
        <li> type == -1: maximum norm (L-infinity): maximum of absolute values of the array elements
        <li> type == 0: count norm (L0): number of non-zero elements
        <li> type == 1: Manhattan norm (L1): sum of absolute values of the array elements
        <li> type == 2: Euclidean norm (L2): square root of <tt>squaredNorm()</tt> when \a useSquaredNorm is <tt>true</tt>,<br>
             or direct algorithm that avoids underflow/overflow otherwise.
        </ul>

        Parameter \a useSquaredNorm has no effect when \a type != 2. Defaults: compute L2 norm as square root of
        <tt>squaredNorm()</tt>.
     */
template <int N, class T>
NormType<ArrayViewND<N, T> >
norm(ArrayViewND<N, T> const & array, int type = 2)
{
    switch(type)
    {
      case -1:
      {
        auto res = NormType<ArrayViewND<N, T> >();
        array_detail::genericArrayFunction(array,
            [&res](T const & v)
            {
                if(res < abs(v))
                    res = abs(v);
            });
        return res;
      }
      case 0:
      {
        auto res = NormType<ArrayViewND<N, T> >();
        auto zero = T();
        array_detail::genericArrayFunction(array,
            [&res, zero](T const & v)
            {
                if(v != zero)
                    res += 1;
            });
        return res;
      }
      case 1:
      {
        auto res = NormType<ArrayViewND<N, T> >();
        array_detail::genericArrayFunction(array,
            [&res](T const & v)
            {
                res += abs(v);
            });
        return res;
      }
      case 2:
      {
        auto res = SquaredNormType<ArrayViewND<N, T> >();
        array_detail::genericArrayFunction(array,
            [&res](T const & v)
            {
                res += v*v;
            });
        return sqrt(res);
      }
      default:
        vigra_precondition(false,
            "norm(ArrayViewND, type): type must be 0, 1, or 2.");
        return NormType<ArrayViewND<N, T> >();
    }
}

template <int N, class T, class U = PromoteType<T> >
inline U
sum(ArrayViewND<N, T> const & array, U init = U{})
{
    return array.template sum<U>(init);
}

template <int N, class T, class U = PromoteType<T> >
inline U
prod(ArrayViewND<N, T> const & array, U init = U{1})
{
    return array.template prod<U>(init);
}

template <int N, class T>
inline bool
all(ArrayViewND<N, T> const & array)
{
    return array.all();
}

template <int N, class T>
inline bool
any(ArrayViewND<N, T> const & array)
{
    return array.any();
}

template <int N, class T>
inline ArrayViewND<N, T>
transpose(ArrayViewND<N, T> const & array)
{
    return array.transpose();
}

template <int N, class T>
inline void
swap(ArrayViewND<N,T> & array1, ArrayViewND<N,T> & array2)
{
    array1.swap(array2);
}

template <int N, class T, class Alloc /* default already declared */ >
class ArrayND
: public ArrayViewND<N, T>
{
    typedef std::vector<typename view_type::value_type, Alloc> DataVector;

    DataVector allocated_data_;

  public:

    typedef ArrayViewND<N, T> view_type;

    using view_type::actual_dimension;

        /** the allocator type used to allocate the memory
         */
    typedef Alloc allocator_type;

        /** the array's value type
         */
    typedef typename view_type::value_type value_type;

        /** pointer type
         */
    typedef typename view_type::pointer pointer;

        /** const pointer type
         */
    typedef typename view_type::const_pointer const_pointer;

        /** reference type (result of operator[])
         */
    typedef typename view_type::reference reference;

        /** const reference type (result of operator[] const)
         */
    typedef typename view_type::const_reference const_reference;

        /** size type
         */
    typedef typename view_type::size_type size_type;

        /** difference type (used for multi-dimensional offsets and indices)
         */
    typedef typename view_type::difference_type difference_type;

        /** difference and index type for a single dimension
         */
    typedef typename view_type::difference_type_1 difference_type_1;

    typedef typename view_type::axistags_type axistags_type;

        /** sequential (random access) iterator type
         */
    typedef typename view_type::iterator iterator;

        /** sequential (random access) const iterator type
         */
    typedef typename view_type::const_iterator const_iterator;

        /** default constructor
         */
    ArrayND()
    {}

        /** construct with given allocator
         */
    explicit
    ArrayND(allocator_type const & alloc)
    : view_type()
    , allocated_data_(alloc)
    {}

        /** Construct with shape given by explicit parameters.

            The number of parameters must match <tt>ndim()</tt>.
         */
    template <class ... V>
    ArrayND(difference_type_1 l0, V ... l)
    : ArrayND(Shape<sizeof...(V)+1>{l0, l...})
    {
        static_assert(N == runtime_size || N == sizeof...(V)+1,
            "ArrayND(int, ...): mismatch between ndim() and number of arguments.");
    }

        /** construct with given shape
         */
    explicit
    ArrayND(difference_type const & shape,
            MemoryOrder order = C_ORDER,
            allocator_type const & alloc = allocator_type())
    : ArrayND(shape, value_type(), order, alloc)
    {}

        /** construct with given shape and axistags
         */
    ArrayND(difference_type const & shape,
            axistags_type const & axistags,
            MemoryOrder order = C_ORDER,
            allocator_type const & alloc = allocator_type())
    : ArrayND(shape, axistags, value_type(), order, alloc)
    {}

        /** construct from shape with an initial value
         */
    ArrayND(difference_type const & shape,
            const_reference init,
            MemoryOrder order = C_ORDER,
            allocator_type const & alloc = allocator_type())
    : view_type(shape, 0, order)
    , allocated_data_(this->size(), init, alloc)
    {
        this->data_  = &allocated_data_[0];
        this->flags_ |= ConsecutiveMemory | OwnsMemory;
    }

        /** construct from shape with an initial value
         */
    ArrayND(difference_type const & shape,
            axistags_type const & axistags,
            const_reference init,
            MemoryOrder order = C_ORDER,
            allocator_type const & alloc = allocator_type())
    : view_type(shape, axistags, 0, order)
    , allocated_data_(this->size(), init, alloc)
    {
        this->data_  = &allocated_data_[0];
        this->flags_ |= ConsecutiveMemory | OwnsMemory;
    }

        // /** construct from shape and initialize with a linear sequence in scan order
            // (i.e. first pixel gets value 0, second on gets value 1 and so on).
         // */
    // ArrayND (const difference_type &shape, MultiArrayInitializationTag init,
                // allocator_type const & alloc = allocator_type());

        /** construct from shape and copy values from the given C array
         */
    ArrayND(difference_type const & shape,
            const_pointer init,
            MemoryOrder order = C_ORDER,
            allocator_type const & alloc = allocator_type())
    : view_type(shape, 0, order)
    , allocated_data_(init, init + this->size(), alloc)
    {
        this->data_  = &allocated_data_[0];
        this->flags_ |= ConsecutiveMemory | OwnsMemory;
    }

        /** construct from shape and and axistags and copy values from the given C array
         */
    ArrayND(difference_type const & shape,
            axistags_type const & axistags,
            const_pointer init,
            MemoryOrder order = C_ORDER,
            allocator_type const & alloc = allocator_type())
    : view_type(shape, axistags, 0, order)
    , allocated_data_(init, init + this->size(), alloc)
    {
        this->data_  = &allocated_data_[0];
        this->flags_ |= ConsecutiveMemory | OwnsMemory;
    }

        /** copy constructor
         */
    ArrayND(ArrayND const & rhs)
    : view_type(rhs)
    , allocated_data_(rhs.allocated_data_)
    {
        this->data_  = &allocated_data_[0];
        this->flags_ |= ConsecutiveMemory | OwnsMemory;
    }

        /** move constructor
         */
    ArrayND(ArrayND && rhs)
    : view_type()
    , allocated_data_(std::move(rhs.allocated_data_))
    {
        this->swapImpl(rhs);
        this->data_  = &allocated_data_[0];
        this->flags_ |= ConsecutiveMemory | OwnsMemory;
    }

        /** construct by copying from a ArrayViewND
         */
    template <int M, class U>
    ArrayND(ArrayViewND<M, U> const & rhs,
            allocator_type const & alloc = allocator_type())
    : view_type(rhs)
    , allocated_data_(alloc)
    {
        allocated_data_.reserve(this->size());

        auto p = array_detail::permutationToOrder(this->shape(),
                                                  this->strides(), C_ORDER);
        array_detail::genericArrayFunction(rhs.transpose(p),
            [&data=allocated_data_](U const & u)
            {
                data.emplace_back(u);
            });

        this->data_  = &allocated_data_[0];
        this->flags_ |= ConsecutiveMemory | OwnsMemory;
    }

        /** constructor from an array expression
         */
    template<class Expression>
    ArrayND(Expression const & rhs,
            MemoryOrder order = C_ORDER,
            allocator_type const & alloc = allocator_type(),
            enable_if_t<ArrayMathConcept<Expression>::value, bool> = true)
    : view_type(rhs.shape(), 0, order)
    , allocated_data_(alloc)
    {
        allocated_data_.reserve(this->size());

        if(order != C_ORDER)
        {
            auto p = array_detail::permutationToOrder(this->shape(), this->strides(), C_ORDER);
            rhs.transpose(p);
        }

        using U = typename Expression::value_type;
        array_detail::genericArrayFunctionImpl(rhs, rhs.shape(),
            [&data=allocated_data_](U const & u)
            {
                data.emplace_back(u);
            });

        this->data_  = &allocated_data_[0];
        this->flags_ |= ConsecutiveMemory | OwnsMemory;
    }

        /** Assignment.<br>
            If the size of \a rhs is the same as the left-hand side arrays's
            old size, only the data are copied. Otherwise, new storage is
            allocated, which invalidates all objects (array views, iterators)
            depending on the lhs array.
         */
    ArrayND & operator=(ArrayND const & rhs)
    {
        if (this != &rhs)
        {
            if(this->shape() == rhs.shape())
                this->copyImpl(rhs);
            else
                ArrayND(rhs).swap(*this);
        }
        return *this;
    }

        /** Move assignment.<br>
            If the size of \a rhs is the same as the left-hand side arrays's
            old size, only the data are copied. Otherwise, the storage of the
            rhs is moved to the lhs, which invalidates all
            objects (array views, iterators) depending on the lhs array.
         */
    ArrayND & operator=(ArrayND && rhs)
    {
        if (this != &rhs)
        {
            if(this->shape() == rhs.shape())
                this->copyImpl(rhs);
            else
                rhs.swap(*this);
        }
        return *this;
    }

        /** Assignment of a scalar.
         */
    ArrayND & operator=(value_type const & u)
    {
        view_type::operator=(u);
        return *this;
    }

#ifdef DOXYGEN
        /** Assignment from arbitrary ARRAY.

            If the left array has no data or the shapes match, it becomes a copy
            of \a rhs. Otherwise, the function fails with an exception.
         */
    template<class ARRAY>
    ArrayND & operator=(ARRAY const & rhs);

        /** Add-assignment of a differently typed array or an array expression.

            The function fails with an exception when the shapes do not match, unless
            the left array has no data (hasData() is false), in which case the function acts as
            a normal assignment.
         */
    template <class ARRAY>
    ArrayND & operator+=(ARRAY const & rhs);

        /** Subtract-assignment of a differently typed array or an array expression.

            The function fails with an exception when the shapes do not match, unless
            the left array has no data (hasData() is false), in which case the function acts as
            a normal assignment.
         */
    template <class ARRAY>
    ArrayND & operator-=(ARRAY const & rhs);

        /** Multiply-assignment of a differently typed array or an array expression.

            The function fails with an exception when the shapes do not match, unless
            the left array has no data (hasData() is false), in which case the function acts as
            a normal assignment.
         */
    template <class ARRAY>
    ArrayND & operator*=(ARRAY const & rhs);

        /** Divide-assignment of a differently typed array or an array expression.

            The function fails with an exception when the shapes do not match, unless
            the left array has no data (hasData() is false), in which case the function acts as
            a normal assignment.
         */
    template <class ARRAY>
    ArrayND & operator/=(ARRAY const & rhs);

#else

    template<class ARG>
    enable_if_t<ArrayNDConcept<ARG>::value || ArrayMathConcept<ARG>::value,
                ArrayND &>
    operator=(ARG const & rhs)
    {
        static_assert(std::is_convertible<typename ARG::value_type, value_type>::value,
            "ArrayND::operator=(ARRAY const &): value_types of lhs and rhs are incompatible.");
        if(this->shape() == rhs.shape())
            view_type::operator=(rhs);
        else
            ArrayND(rhs).swap(*this);
        return *this;
    }

#define VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(OP) \
    template <class ARG> \
    enable_if_t<ArrayNDConcept<ARG>::value || ArrayMathConcept<ARG>::value, \
                ArrayND &> \
    operator OP(ARG const & rhs) \
    { \
        static_assert(std::is_convertible<typename ARG::value_type, value_type>::value, \
            "ArrayND::operator" #OP "(ARRAY const &): value_types of lhs and rhs are incompatible."); \
        if(this->hasData()) \
            view_type::operator OP(rhs); \
        else \
            ArrayND(rhs).swap(*this); \
        return *this; \
    }

    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(+=)
    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(-=)
    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(*=)
    VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT(/=)

#undef VIGRA_ARRAYND_ARITHMETIC_ASSIGNMENT

#endif  // DOXYGEN

        /** Add-assignment of a scalar.
         */
    ArrayND & operator+=(value_type const & u)
    {
        view_type::operator+=(u);
        return *this;
    }

        /** Subtract-assignment of a scalar.
         */
    ArrayND & operator-=(value_type const & u)
    {
        view_type::operator-=(u);
        return *this;
    }

        /** Multiply-assignment of a scalar.
         */
    ArrayND & operator*=(value_type const & u)
    {
        view_type::operator*=(u);
        return *this;
    }

        /** Divide-assignment of a scalar.
         */
    ArrayND & operator/=(value_type const & u)
    {
        view_type::operator/=(u);
        return *this;
    }

    void
    resize(difference_type const & new_shape,
           axistags_type const & new_axistags = axistags_type{},
           MemoryOrder order = C_ORDER)
    {
        if(this->size() == vigra::prod(new_shape))
        {
            this->swapImpl(this->reshape(new_shape, new_axistags, order));
            this->flags_ |= OwnsMemory;
        }
        else
        {
            ArrayND(new_shape, new_axistags, order).swap(*this);
        }
    }

    template <int M>
    void
    resize(difference_type const & new_shape,
           MemoryOrder order)
    {
        resize(new_shape, axistags_type{}, order);
    }

    void swap(ArrayND & rhs)
    {
        this->swapImpl(rhs);
        allocated_data_.swap(rhs.allocated_data_);
    }

        /** get the allocator.
         */
    allocator_type const & allocator() const
    {
        return allocated_data_.allocator();
    }
};

template <int N, class T, class A>
inline void
swap(ArrayND<N,T,A> & array1, ArrayND<N,T,A> & array2)
{
    array1.swap(array2);
}

namespace array_detail {



template <class HANDLE, int N, class T, class ... REST>
auto
makeCoupledIteratorImpl(MemoryOrder order, HANDLE const & inner_handle,
                        ArrayViewND<N, T> const & a, REST const & ... rest)
    -> decltype(makeCoupledIteratorImpl(order, *(HandleNDChain<T, HANDLE>*)0, rest...))
{
    static_assert(CompatibleDimensions<N, HANDLE::dimension>::value,
        "makeCoupledIterator(): arrays have incompatible dimensions.");
    vigra_precondition(a.shape() == inner_handle.shape(),
        "makeCoupledIterator(): arrays have incompatible shapes.");
    HandleNDChain<T, HANDLE> handle(a.handle(), inner_handle);
    return makeCoupledIteratorImpl(order, handle, rest ...);
}

template <class HANDLE>
IteratorND<HANDLE>
makeCoupledIteratorImpl(MemoryOrder order, HANDLE const & handle)
{
    return IteratorND<HANDLE>(handle, order);
}

} // namespace array_detail

template <int N, class T, class ... REST>
auto
makeCoupledIterator(ArrayViewND<N, T> const & a, REST const & ... rest)
    -> decltype(array_detail::makeCoupledIteratorImpl(C_ORDER, *(HandleNDChain<T, ShapeHandle<N>>*)0, rest...))
{
    HandleNDChain<T, ShapeHandle<N>> handle(a.handle(), ShapeHandle<N>(a.shape()));
    return array_detail::makeCoupledIteratorImpl(C_ORDER, handle, rest ...);
}

template <int N, class T, class ... REST>
auto
makeCoupledIterator(MemoryOrder order, ArrayViewND<N, T> const & a, REST const & ... rest)
    -> decltype(array_detail::makeCoupledIteratorImpl(order, *(HandleNDChain<T, ShapeHandle<N>>*)0, rest...))
{
    HandleNDChain<T, ShapeHandle<N>> handle(a.handle(), ShapeHandle<N>(a.shape()));
    return array_detail::makeCoupledIteratorImpl(order, handle, rest ...);
}

} // namespace vigra

#endif // VIGRA2_ARRAY_ND_HXX
