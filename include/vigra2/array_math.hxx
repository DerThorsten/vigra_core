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

#ifndef VIGRA2_ARRAY_MATH_HXX
#define VIGRA2_ARRAY_MATH_HXX

#include "metaprogramming.hxx"
#include "numeric_traits.hxx"
#include "mathutil.hxx"
#include "tinyarray.hxx"
#include "pointer_nd.hxx"
#include <vector>
#include <type_traits>

namespace vigra {

/** \defgroup ArrayMathModule vigra::array_math

    Namespace <tt>vigra::array_math</tt> holds VIGRA's support for efficient arithmetic and algebraic functions on multi-dimensional arrays (that is, \ref MultiArrayView and its subclasses). All <tt>array_math</tt> functions operate element-wise. If you need matrix multiplication, use \ref LinearAlgebraModule instead.

    In order to avoid overload ambiguities, multi-array arithmetic must be explicitly activated by
    \code
    using namespace vigra::array_math;
    \endcode
    (this should not be done globally, but only in the scope where the functionality is actually used).

    You can then use the standard operators in the expected way:
    \code
    MultiArray<2, float> i(Shape2(100, 100)), j(Shape2(100, 100));

    MultiArray<2, float> h  = i + 4.0 * j;
                         h += (i.transpose() - j) / 2.0;
    \endcode
    etc. (supported operators are <tt>+ - * / ! ~ % && || == != &lt; &lt;= &gt; &gt;= &lt;&lt; &gt;&gt; & | ^ = += -= *= /=</tt>, with both scalar and array arguments).

    Algebraic functions are available as well:
    \code
    h  = exp(-(sq(i) + sq(j)));
    h *= atan2(-i, j);
    \endcode
    The following functions are implemented: <tt>abs, erf, even, odd, sign, signi, round, roundi, sqrt, sqrti, sq,
    norm, squaredNorm, gamma, loggamma, exp, log, log10, sin, sin_pi, cos, cos_pi, asin, acos, tan, atan,
    floor, ceil, conj, real, imag, arg, atan2, pow, fmod, min, max</tt>,
    provided the array's element type supports the respective function.

    Supported element types currently include the built-in numeric types, \ref TinyVector, \ref RGBValue,
    <tt>std::complex</tt>, and \ref FFTWComplex.

    In addition, <tt>array_math</tt> supports a number of functions that reduce arrays to scalars:
    \code
    double s = sum<double>(i);  // compute the sum of the elements, using 'double' as accumulator type
    double p = product<double>(abs(i));  // compute the product of the elements' absolute values

    bool a = any(i < 0.0);  // check if any element of i is negative
    bool b = all(i > 0.0);  // check if all elements of i are positive
    \endcode

    Expressions are expanded so that no temporary arrays have to be created. To optimize cache locality,
    loops are executed in the stride ordering of the left-hand-side array.

    <b>\#include</b> \<vigra/array_math.hxx\>

    Namespace: vigra::array_math
*/
namespace array_math {

using vigra::array_detail::MemoryOverlap;

/********************************************************/
/*                                                      */
/*               ArrayMathUnifyDimension                */
/*                                                      */
/********************************************************/

    // Compute the common ndim for two arrays.
template <class A1, class A2>
struct ArrayMathUnifyDimension
{
    static const int M = A1::dimension;
    static const int N = A2::dimension;

    typedef integral_minmax<N, M>  minmax;

    static_assert(minmax::min <= 0 || N == M,
        "array_math: incompatible array dimensions.");

    static const int value = minmax::max > 0
                                ? minmax::max
                                : minmax::min;
};

/********************************************************/
/*                                                      */
/*                 ArrayMathExpression                  */
/*                                                      */
/********************************************************/

    // Basic template for array expression templates
template <class ARG>
struct ArrayMathExpression
: public ARG
{
    static_assert(ArrayMathConcept<ARG>::value,
        "ArrayMathExpression<ARG>: ARG must fulfill the ArrayMathConcept.");

    using ARG::ARG;

    ArrayMathExpression & pointer_nd() const
    {
        return *this;
    }

    ArrayMathExpression pointer_nd()
    {
        return ArrayMathExpression(*this);
    }

    template <class SHAPE>
    ArrayMathExpression &
    pointer_nd(SHAPE const & permutation)
    {
        this->transpose_inplace(permutation);
        return *this;
    }

    template <class SHAPE>
    ArrayMathExpression
    pointer_nd(SHAPE const & permutation) const
    {
        ArrayMathExpression res(*this);
        res.transpose_inplace(permutation);
        return res;
    }
};

/********************************************************/
/*                                                      */
/*         ArrayMathExpression<PointerND<0, T>>         */
/*                                                      */
/********************************************************/

    // Class to represent constants in array expressions. It inherits
    // PointerND<0, T> so that we can later call universalPointerNDFunction()
    // to do the actual computations.
template <class T>
struct ArrayMathExpression<PointerND<0, T>>
: public PointerND<0, T>
, public ArrayMathTag
{
    typedef PointerND<0, T>                      base_type;
    typedef typename base_type::difference_type  difference_type;

    difference_type shape_;

    ArrayMathExpression(T const & data)
    : base_type(data)
    {}

    constexpr MemoryOverlap checkMemoryOverlap(TinyArray<char*, 2> const &) const
    {
        return vigra::array_detail::NoMemoryOverlap;
    }

    template <class SHAPE>
    void principalStrides(SHAPE const &, ArrayIndex, int) const
    {}

    template <class SHAPE>
    void transpose_inplace(SHAPE const & permutation)
    {
        if (this->ndim() == permutation.size())
            shape_ = shape_.transpose(permutation);
    }

    void setShape(difference_type shape)
    {
        shape_ = std::move(shape);
    }

    difference_type const & shape() const
    {
        return shape_;
    }

    template <class SHAPE>
    bool unifyShape(SHAPE &) const
    {
        return true;
    }
};

/********************************************************/
/*                                                      */
/*          ArrayMathExpression<PointerND<N, T>>        */
/*                                                      */
/********************************************************/

    // Class to represent arrays in array expressions. It inherits the appropriate
    // PointerND<N, T> so that we can later call universalPointerNDFunction()
    // to do the actual computations.
template <int N, class T>
struct ArrayMathExpression<PointerND<N, T>>
: public PointerND<N, T>
, public ArrayMathTag
{
    typedef PointerND<N, T>                      base_type;
    typedef typename base_type::difference_type  difference_type;

    difference_type shape_;

    ArrayMathExpression(ArrayViewND<N, T> const & array)
    : base_type(const_cast<ArrayViewND<N, T> &>(array).pointer_nd())
    , shape_(array.shape())
    {}

    ArrayMathExpression(base_type const & p, difference_type const & shape)
    : base_type(p)
    , shape_(shape)
    {}

    TinyArray<char *, 2> memoryRange() const
    {
        return { this->data_, (char*)(1 + &(*this)[shape_ - 1]) };
    }

    MemoryOverlap checkMemoryOverlap(TinyArray<char*, 2> const & target) const
    {
        return vigra::array_detail::checkMemoryOverlap(target, memoryRange());
    }

    template <class SHAPE>
    void principalStrides(SHAPE & strides, ArrayIndex & minimalStride, int & singletonCount) const
    {
        array_detail::principalStrides(strides, this->strides_, minimalStride, singletonCount);
    }

    template <class SHAPE>
    void principalStrides(SHAPE & strides) const
    {
        ArrayIndex minimalStride = NumericTraits<ArrayIndex>::max();
        int singletonCount       = this->ndim();
        principalStrides(strides, minimalStride, singletonCount);
    }

    template <class SHAPE>
    void transpose_inplace(SHAPE const & permutation)
    {
        this->strides_ = this->strides_.transpose(permutation);
        shape_ = shape_.transpose(permutation);
    }

    difference_type const & shape() const
    {
        return shape_;
    }

    template <class SHAPE>
    bool unifyShape(SHAPE & target) const
    {
        return vigra::detail::unifyShape(target, shape_);
    }
};

/********************************************************/
/*                                                      */
/*                   ArrayMathArgType                   */
/*                                                      */
/********************************************************/

    // Choose the appropriate ArrayMathExpression according to the ARG type.
template <class ARG>
struct ArrayMathTypeChooser
{
    typedef typename
        std::conditional<ArrayMathConcept<ARG>::value,
                         ARG,
                         ArrayMathExpression<PointerND<0, ARG>>>::type type;
};

template <int N, class T>
struct ArrayMathTypeChooser<ArrayViewND<N, T>>
{
    typedef ArrayMathExpression<PointerND<N, T>> type;
};

template <int N, class T, class A>
struct ArrayMathTypeChooser<ArrayND<N, T, A>>
    : public ArrayMathTypeChooser<ArrayViewND<N, T>>
{};

template <class ARG>
using ArrayMathArgType = typename ArrayMathTypeChooser<ARG>::type;

/********************************************************/
/*                                                      */
/*                ArrayMathUnaryOperator                */
/*                                                      */
/********************************************************/

    // Base class for unary operators/functions in array expressions.
    // It implements the PointerND API so that we can later call
    // universalPointerNDFunction() to do the actual computations.
template <class ARG>
struct ArrayMathUnaryOperator
: public ArrayMathTag
{
    typedef ArrayMathArgType<ARG> arg_type;

    typedef typename arg_type::difference_type  difference_type;
    static const int dimension                = arg_type::dimension;

    arg_type arg_;

    ArrayMathUnaryOperator(ARG const & a)
    : arg_(a)
    {}

    bool hasData() const
    {
        return arg_.hasData();
    }

    MemoryOverlap checkMemoryOverlap(TinyArray<char*, 2> const & target) const
    {
        return arg_.checkMemoryOverlap(target);
    }

    template <class SHAPE>
    bool compatibleStrides(SHAPE const & target) const
    {
        return arg_.compatibleStrides(target);
    }

    template <class SHAPE>
    void principalStrides(SHAPE & strides, ArrayIndex & minimalStride, int & singletonCount) const
    {
        arg_.principalStrides(strides, minimalStride, singletonCount);
    }

    template <class SHAPE>
    void principalStrides(SHAPE & strides) const
    {
        ArrayIndex minimalStride = NumericTraits<ArrayIndex>::max();
        int singletonCount = ndim();
        principalStrides(strides, minimalStride, singletonCount);
    }

    template <class SHAPE>
    void transpose_inplace(SHAPE const & permutation)
    {
        arg_.transpose_inplace(permutation);
    }

    // increment the pointer of all RHS arrays along the given 'axis'
    void inc(int axis)
    {
        arg_.inc(axis);
    }

    // decrement the pointer of all RHS arrays along the given 'axis'
    void dec(int axis)
    {
        arg_.dec(axis);
    }

    // reset the pointer of all RHS arrays along the given 'axis'
    void move(int axis, ArrayIndex diff)
    {
        arg_.move(axis, diff);
    }

    difference_type shape() const
    {
        return arg_.shape();
    }

    template <int M = dimension>
    int ndim(enable_if_t<M == runtime_size, bool> = true) const
    {
        return arg_.ndim();
    }

    template <int M = dimension>
    constexpr
        int ndim(enable_if_t<M != runtime_size, bool> = true) const
    {
        return dimension;
    }

    template <class SHAPE>
    bool unifyShape(SHAPE & target) const
    {
        return arg_.unifyShape(target);
    }
};

/********************************************************/
/*                                                      */
/*           unary array functions/operators            */
/*                                                      */
/********************************************************/

#define VIGRA_ARRAY_MATH_UNARY(NAME, CALL, FUNCTION) \
template <class ARG> \
struct ArrayMath##NAME \
: public ArrayMathUnaryOperator<ARG> \
{ \
    typedef ArrayMathUnaryOperator<ARG>      base_type; \
    typedef typename base_type::arg_type     arg_type; \
    typedef decltype(CALL(**(arg_type*)0))   raw_result_type; \
    typedef BoolPromote<raw_result_type>     result_type; \
    typedef typename std::remove_reference<result_type>::type   value_type; \
 \
    ArrayMath##NAME(ARG const & a) \
    : base_type(a) \
    {} \
 \
    value_type const * ptr() const { return 0; } \
 \
    result_type operator*() const \
    { \
        return CALL(*this->arg_); \
    } \
 \
    template <class SHAPE> \
    result_type operator[](SHAPE const & s) const \
    { \
        return CALL(this->arg_[s]); \
    } \
}; \
 \
template <class ARG> \
enable_if_t<ArrayNDConcept<ARG>::value || ArrayMathConcept<ARG>::value, \
            ArrayMathExpression<ArrayMath##NAME<ARG>>> \
FUNCTION(ARG const & arg) \
{ \
    return {arg}; \
}

VIGRA_ARRAY_MATH_UNARY(Negate, -, operator-)
VIGRA_ARRAY_MATH_UNARY(Not, !, operator!)
VIGRA_ARRAY_MATH_UNARY(BitwiseNot, ~, operator~)

VIGRA_ARRAY_MATH_UNARY(Abs, vigra::abs, abs)
VIGRA_ARRAY_MATH_UNARY(Fabs, vigra::fabs, fabs)

VIGRA_ARRAY_MATH_UNARY(Cos, vigra::cos, cos)
VIGRA_ARRAY_MATH_UNARY(Sin, vigra::sin, sin)
VIGRA_ARRAY_MATH_UNARY(Tan, vigra::tan, tan)
VIGRA_ARRAY_MATH_UNARY(Sin_pi, vigra::sin_pi, sin_pi)
VIGRA_ARRAY_MATH_UNARY(Cos_pi, vigra::cos_pi, cos_pi)
VIGRA_ARRAY_MATH_UNARY(Acos, vigra::acos, acos)
VIGRA_ARRAY_MATH_UNARY(Asin, vigra::asin, asin)
VIGRA_ARRAY_MATH_UNARY(Atan, vigra::atan, atan)

VIGRA_ARRAY_MATH_UNARY(Cosh, vigra::cosh, cosh)
VIGRA_ARRAY_MATH_UNARY(Sinh, vigra::sinh, sinh)
VIGRA_ARRAY_MATH_UNARY(Tanh, vigra::tanh, tanh)
VIGRA_ARRAY_MATH_UNARY(Acosh, vigra::acosh, acosh)
VIGRA_ARRAY_MATH_UNARY(Asinh, vigra::asinh, asinh)
VIGRA_ARRAY_MATH_UNARY(Atanh, vigra::atanh, atanh)

VIGRA_ARRAY_MATH_UNARY(Sqrt, vigra::sqrt, sqrt)
VIGRA_ARRAY_MATH_UNARY(Cbrt, vigra::cbrt, cbrt)
VIGRA_ARRAY_MATH_UNARY(Sqrti, vigra::sqrti, sqrti)
VIGRA_ARRAY_MATH_UNARY(Sq, vigra::sq, sq)
VIGRA_ARRAY_MATH_UNARY(Norm, vigra::norm, elementwiseNorm)
VIGRA_ARRAY_MATH_UNARY(SquaredNorm, vigra::squaredNorm, elementwiseSquaredNorm)

VIGRA_ARRAY_MATH_UNARY(Exp, vigra::exp, exp)
VIGRA_ARRAY_MATH_UNARY(Exp2, vigra::exp2, exp2)
VIGRA_ARRAY_MATH_UNARY(Expm1, vigra::expm1, expm1)
VIGRA_ARRAY_MATH_UNARY(Log, vigra::log, log)
VIGRA_ARRAY_MATH_UNARY(Log2, vigra::log2, log2)
VIGRA_ARRAY_MATH_UNARY(Log10, vigra::log10, log10)
VIGRA_ARRAY_MATH_UNARY(Log1p, vigra::log1p, log1p)
VIGRA_ARRAY_MATH_UNARY(Logb, vigra::logb, logb)
VIGRA_ARRAY_MATH_UNARY(Ilogb, vigra::ilogb, ilogb)

VIGRA_ARRAY_MATH_UNARY(Ceil, vigra::ceil, ceil)
VIGRA_ARRAY_MATH_UNARY(Floor, vigra::floor, floor)
VIGRA_ARRAY_MATH_UNARY(Trunc, vigra::trunc, trunc)
VIGRA_ARRAY_MATH_UNARY(Round, vigra::round, round)
VIGRA_ARRAY_MATH_UNARY(Lround, vigra::lround, lround)
VIGRA_ARRAY_MATH_UNARY(Llround, vigra::llround, llround)
VIGRA_ARRAY_MATH_UNARY(Roundi, vigra::roundi, roundi)
VIGRA_ARRAY_MATH_UNARY(Even, vigra::even, even)
VIGRA_ARRAY_MATH_UNARY(Odd, vigra::odd, odd)
VIGRA_ARRAY_MATH_UNARY(Sign, vigra::sign, sign)
VIGRA_ARRAY_MATH_UNARY(Signi, vigra::signi, signi)

VIGRA_ARRAY_MATH_UNARY(Erf, vigra::erf, erf)
VIGRA_ARRAY_MATH_UNARY(Erfc, vigra::erfc, erfc)
VIGRA_ARRAY_MATH_UNARY(Tgamma, vigra::tgamma, tgamma)
VIGRA_ARRAY_MATH_UNARY(Lgamma, vigra::lgamma, lgamma)
VIGRA_ARRAY_MATH_UNARY(Loggamma, vigra::lgamma, loggamma)

VIGRA_ARRAY_MATH_UNARY(Conj, conj, conj)
VIGRA_ARRAY_MATH_UNARY(Real, real, real)
VIGRA_ARRAY_MATH_UNARY(Imag, imag, imag)
VIGRA_ARRAY_MATH_UNARY(Arg, arg, arg)

#undef VIGRA_ARRAY_MATH_UNARY

/********************************************************/
/*                                                      */
/*                ArrayMathBinaryOperator               */
/*                                                      */
/********************************************************/

    // Base class for binary operators/functions in array expressions.
    // It implements the PointerND API so that we can later call
    // universalPointerNDFunction() to do the actual computations.
template <class ARG1, class ARG2>
struct ArrayMathBinaryOperator
: public ArrayMathTag
{
    typedef ArrayMathArgType<ARG1> arg1_type;
    typedef ArrayMathArgType<ARG2> arg2_type;
    static const int dimension = ArrayMathUnifyDimension<arg1_type, arg2_type>::value;
    typedef Shape<dimension> difference_type;

    arg1_type arg1_;
    arg2_type arg2_;

    ArrayMathBinaryOperator(ARG1 const & a1, ARG2 const & a2)
    : arg1_(a1)
    , arg2_(a2)
    {}

    bool hasData() const
    {
        return arg1_.hasData() && arg2_.hasData();
    }

    MemoryOverlap checkMemoryOverlap(TinyArray<char*, 2> const & target) const
    {
        return (MemoryOverlap)(arg1_.checkMemoryOverlap(target) | arg2_.checkMemoryOverlap(target));
    }

    template <class SHAPE>
    bool compatibleStrides(SHAPE const & target) const
    {
        return arg1_.compatibleStrides(target) && arg2_.compatibleStrides(target);
    }

    template <class SHAPE>
    void principalStrides(SHAPE & strides, ArrayIndex & minimalStride, int & singletonCount) const
    {
        arg1_.principalStrides(strides, minimalStride, singletonCount);
        arg2_.principalStrides(strides, minimalStride, singletonCount);
    }

    template <class SHAPE>
    void principalStrides(SHAPE & strides) const
    {
        ArrayIndex minimalStride = NumericTraits<ArrayIndex>::max();
        int singletonCount = ndim();
        principalStrides(strides, minimalStride, singletonCount);
    }

    template <class SHAPE>
    void transpose_inplace(SHAPE const & permutation)
    {
        arg1_.transpose_inplace(permutation);
        arg2_.transpose_inplace(permutation);
    }

    // increment the pointer of all RHS arrays along the given 'axis'
    void inc(int axis)
    {
        arg1_.inc(axis);
        arg2_.inc(axis);
    }

    // decrement the pointer of all RHS arrays along the given 'axis'
    void dec(int axis)
    {
        arg1_.dec(axis);
        arg2_.dec(axis);
    }

    // reset the pointer of all RHS arrays along the given 'axis'
    void move(int axis, ArrayIndex diff)
    {
        arg1_.move(axis, diff);
        arg2_.move(axis, diff);
    }

    difference_type shape() const
    {
        Shape<dimension> res(tags::size = ndim(), 1);
        vigra_precondition(unifyShape(res),
            "ArrayMathBinaryOperator(): shape mismatch.");
        return res;
    }

    template <int M = dimension>
    int ndim(enable_if_t<M == runtime_size, bool> = true) const
    {
        return max(arg1_.ndim(), arg2_.ndim());
    }

    template <int M = dimension>
    constexpr
    int ndim(enable_if_t<M != runtime_size, bool> = true) const
    {
        return dimension;
    }

    template <class SHAPE>
    bool unifyShape(SHAPE & target) const
    {
        return arg1_.unifyShape(target) && arg2_.unifyShape(target);
    }
};

/********************************************************/
/*                                                      */
/*                 ArrayMathBinaryTraits                */
/*                                                      */
/********************************************************/

    // Determine if an array expression should be created for the given args.
    // If neither arg is an array or array expression, the answer is always "no".
    // Otherwise, the answer is "yes" if the respective value types are compatible.
template <class ARG1, class ARG2,
          bool IS_ARRAY1 = ArrayNDConcept<ARG1>::value ||
                           ArrayMathConcept<ARG1>::value,
          bool IS_ARRAY2 = ArrayNDConcept<ARG2>::value ||
                           ArrayMathConcept<ARG2>::value>
struct ArrayMathBinaryTraits
{
    typedef PromoteTraits<typename ARG1::value_type,
                          typename ARG2::value_type> Compatible;
    static const bool value = Compatible::value;
};


template <class ARG1, class ARG2>
struct ArrayMathBinaryTraits<ARG1, ARG2, true, false>
{
    typedef PromoteTraits<typename ARG1::value_type,
                          ARG2> Compatible;
    static const bool value = Compatible::value;
};

template <class ARG1, class ARG2>
struct ArrayMathBinaryTraits<ARG1, ARG2, false, true>
{
    typedef PromoteTraits<ARG1,
                          typename ARG2::value_type> Compatible;
    static const bool value = Compatible::value;
};

template <class ARG1, class ARG2>
struct ArrayMathBinaryTraits<ARG1, ARG2, false, false>
{
    static const bool value = false;
};

/********************************************************/
/*                                                      */
/*           binary array functions/operators           */
/*                                                      */
/********************************************************/

#define VIGRA_ARRAYMATH_BINARY_OPERATOR(NAME, CALL, FUNCTION, SEP) \
\
template <class ARG1, class ARG2> \
struct ArrayMath##NAME \
: public ArrayMathBinaryOperator<ARG1, ARG2> \
{ \
    typedef ArrayMathBinaryOperator<ARG1, ARG2>   base_type; \
    typedef typename base_type::arg1_type         arg1_type; \
    typedef typename base_type::arg2_type         arg2_type; \
    typedef typename arg1_type::value_type        value1_type; \
    typedef typename arg2_type::value_type        value2_type; \
    typedef decltype(CALL(*(value1_type*)0 SEP *(value2_type*)0)) raw_result_type; \
    typedef BoolPromote<raw_result_type>          result_type; \
    typedef typename std::remove_reference<result_type>::type   value_type; \
 \
    ArrayMath##NAME(ARG1 const & a1, ARG2 const & a2) \
    : base_type(a1, a2) \
    {} \
 \
    value_type const * ptr() const { return 0; } \
 \
    result_type operator*() const \
    { \
        return CALL(*this->arg1_ SEP *this->arg2_); \
    } \
 \
    template <class SHAPE> \
    result_type operator[](SHAPE const & s) const \
    { \
        return CALL(this->arg1_[s] SEP this->arg2_[s]); \
    } \
}; \
 \
template <class ARG1, class ARG2> \
enable_if_t<ArrayMathBinaryTraits<ARG1, ARG2>::value, \
            ArrayMathExpression<ArrayMath##NAME<ARG1, ARG2>>> \
FUNCTION(ARG1 const & a1, ARG2 const & a2) \
{ \
    return {a1, a2}; \
}

#define VIGRA_NOTHING
#define VIGRA_COMMA ,

VIGRA_ARRAYMATH_BINARY_OPERATOR(Plus, VIGRA_NOTHING, operator+, +)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Minus, VIGRA_NOTHING, operator-, -)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Multiplies, VIGRA_NOTHING, operator*, *)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Divides, VIGRA_NOTHING, operator/, /)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Modulo, VIGRA_NOTHING, operator%, %)
VIGRA_ARRAYMATH_BINARY_OPERATOR(And, VIGRA_NOTHING, operator&&, &&)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Or, VIGRA_NOTHING, operator||, ||)
VIGRA_ARRAYMATH_BINARY_OPERATOR(ElementwiseEqual, VIGRA_NOTHING, elementwiseEqual, ==)
VIGRA_ARRAYMATH_BINARY_OPERATOR(ElementwiseNotEqual, VIGRA_NOTHING, elementwiseNotEqual, !=)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Less, VIGRA_NOTHING, operator<, <)
VIGRA_ARRAYMATH_BINARY_OPERATOR(LessEqual, VIGRA_NOTHING, operator<=, <=)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Greater, VIGRA_NOTHING, operator>, >)
VIGRA_ARRAYMATH_BINARY_OPERATOR(GreaterEqual, VIGRA_NOTHING, operator>=, >=)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Leftshift, VIGRA_NOTHING, operator<<, <<)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Rightshift, VIGRA_NOTHING, operator>>, >>)
VIGRA_ARRAYMATH_BINARY_OPERATOR(BitwiseAnd, VIGRA_NOTHING, operator&, &)
VIGRA_ARRAYMATH_BINARY_OPERATOR(BitwiseOr, VIGRA_NOTHING, operator|, |)
VIGRA_ARRAYMATH_BINARY_OPERATOR(BitwiseXor, VIGRA_NOTHING, operator^, ^)

VIGRA_ARRAYMATH_BINARY_OPERATOR(Atan2, vigra::atan2, atan2, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Copysign, vigra::copysign, copysign, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Fdim, vigra::fdim, fdim, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Fmax, vigra::fmax, fmax, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Fmin, vigra::fmin, fmin, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Fmod, vigra::fmod, fmod, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Hypot, vigra::hypot, hypot, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Pow, vigra::pow, pow, VIGRA_COMMA)

VIGRA_ARRAYMATH_BINARY_OPERATOR(ClipUpper, vigra::clipUpper, clipUpper, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(ClipLower, vigra::clipLower, clipLower, VIGRA_COMMA)

#undef VIGRA_NOTHING
#undef VIGRA_COMMA
#undef VIGRA_ARRAYMATH_BINARY_OPERATOR

#define VIGRA_ARRAYMATH_MINMAX_FUNCTION(NAME, CALL) \
\
template <class ARG1, class ARG2> \
struct ArrayMath##NAME \
: public ArrayMathBinaryOperator<ARG1, ARG2> \
{ \
    typedef ArrayMathBinaryOperator<ARG1, ARG2>   base_type; \
    typedef typename base_type::arg1_type         arg1_type; \
    typedef typename base_type::arg2_type         arg2_type; \
    typedef typename arg1_type::value_type        first_argument_type; \
    typedef typename arg2_type::value_type        second_argument_type; \
 \
    typedef typename \
       std::common_type<first_argument_type, second_argument_type>::type raw_result_type; \
    typedef BoolPromote<raw_result_type> result_type; \
    typedef typename std::remove_reference<result_type>::type   value_type; \
 \
    ArrayMath##NAME(ARG1 const & a1, ARG2 const & a2) \
    : base_type(a1, a2) \
    {} \
 \
    value_type const * ptr() const { return 0; } \
 \
    result_type operator*() const \
    { \
        return CALL(*this->arg1_, *this->arg2_); \
    } \
 \
    template <class SHAPE> \
    result_type operator[](SHAPE const & s) const \
    { \
        return CALL(this->arg1_[s], this->arg2_[s]); \
    } \
}; \
 \
template <class ARG1, class ARG2> \
enable_if_t<ArrayMathBinaryTraits<ARG1, ARG2>::value, \
            ArrayMathExpression<ArrayMath##NAME<ARG1, ARG2>>> \
CALL(ARG1 const & a1, ARG2 const & a2) \
{ \
    return {a1, a2}; \
}

using vigra::min;
using vigra::max;
VIGRA_ARRAYMATH_MINMAX_FUNCTION(Min, min)
VIGRA_ARRAYMATH_MINMAX_FUNCTION(Max, max)

#undef VIGRA_ARRAYMATH_MINMAX_FUNCTION

/********************************************************/
/*                                                      */
/*                     ArrayMathMGrid                   */
/*                                                      */
/********************************************************/

    // Class to wrap a shape object in an array expression. It serves
    // the role of a Matlab meshgrid / Python mgrid.
template <int N>
class ArrayMathExpression<Shape<N>>
: public ArrayMathTag
, public PointerNDShape<N>
{
public:
    typedef PointerNDShape<N>              base_type;
    typedef Shape<N>                       shape_type;

    explicit ArrayMathExpression(shape_type const & shape)
    : base_type(shape)
    {}

    template <class SHAPE>
    constexpr bool compatibleStrides(SHAPE const &) const
    {
        return true;
    }

    inline void inc(int dim)
    {
        base_type::inc(dim);
    }

    inline void dec(int dim)
    {
        base_type::dec(dim);
    }

    void move(int dim, ArrayIndex diff)
    {
        base_type::move(dim, diff);
    }

    template <class SHAPE>
    void transpose_inplace(SHAPE const & permutation)
    {
        this->shape_ = this->shape_.transpose(permutation);
        this->point_ = this->point_.transpose(permutation);
    }

    template <class SHAPE>
    bool unifyShape(SHAPE & target) const
    {
        return vigra::detail::unifyShape(target, this->shape());
    }
};

template <int N>
ArrayMathExpression<Shape<N>>
mgrid(Shape<N> const & shape)
{
    return ArrayMathExpression<Shape<N>>(shape);
}

/********************************************************/
/*                                                      */
/*                 ArrayMathCustomFunctor               */
/*                                                      */
/********************************************************/

template <class ARG1, class ARG2, class FCT>
struct ArrayMathCustomFunctor
: public ArrayMathBinaryOperator<ARG1, ARG2>
{
    typedef ArrayMathBinaryOperator<ARG1, ARG2>   base_type;
    typedef typename base_type::arg1_type         arg1_type;
    typedef typename base_type::arg2_type         arg2_type;
    typedef typename arg1_type::value_type        value1_type;
    typedef typename arg2_type::value_type        value2_type;
    typedef typename std::result_of<FCT(value1_type,value2_type)>::type raw_result_type;
    typedef BoolPromote<raw_result_type>          result_type;
    typedef typename std::remove_reference<result_type>::type   value_type;

    FCT f_;

    ArrayMathCustomFunctor(ARG1 const & a1, ARG2 const & a2, FCT && f)
    : base_type(a1, a2)
    , f_(std::forward<FCT>(f))
    {}

    value_type const * ptr() const { return 0; }

    result_type operator*() const
    {
        return f_(*this->arg1_, *this->arg2_);
    }

    template <class SHAPE>
    result_type operator[](SHAPE const & s) const
    {
        return f_(this->arg1_[s], this->arg2_[s]);
    }
};

} // namespace array_math

using array_math::ArrayMathExpression;
using array_math::min;
using array_math::max;

/********************************************************/
/*                                                      */
/*                 reducing operations                  */
/*                                                      */
/********************************************************/

    // Functions to reduce an array or array expression to a single
    // number: all, all_finite, any, sum, prod, ==, !=

template <class ARG,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline bool
all(ARG && a)
{
    typedef typename ARG::value_type value_type;
    bool res = true;
    value_type zero = value_type();
    universalArrayNDFunction(std::forward<ARG>(a),
        [zero, &res](value_type const & v)
        {
            if (v == zero)
                res = false;
        },
        "all(ARRAY_EXPRESSION)"
    );
    return res;
}

template <class ARG,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline bool
all_finite(ARG && a)
{
    typedef typename ARG::value_type value_type;
    bool res = true;
    universalArrayNDFunction(std::forward<ARG>(a),
        [&res](value_type const & v)
        {
            if (!isfinite(v))
                res = false;
        },
        "all_finite(ARRAY_EXPRESSION)"
    );
    return res;
}

template <class ARG,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline bool
any(ARG && a)
{
    typedef typename ARG::value_type value_type;
    bool res = false;
    value_type zero = value_type();
    universalArrayNDFunction(std::forward<ARG>(a),
        [zero, &res](value_type const & v)
        {
            if (v != zero)
                res = true;
        },
        "any(ARRAY_EXPRESSION)"
    );
    return res;
}

template <class ARG,
          class U = PromoteType<typename ARG::value_type>,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline U
sum(ARG && a, U res = {})
{
    typedef typename ARG::value_type value_type;
    universalArrayNDFunction(std::forward<ARG>(a),
        [&res](value_type const & v)
        {
            res += v;
        },
        "sum(ARRAY_EXPRESSION)"
    );
    return res;
}

template <class ARG,
          class U = PromoteType<typename ARG::value_type>,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline U
prod(ARG && a, U res = U{1})
{
    typedef typename ARG::value_type value_type;
    universalArrayNDFunction(std::forward<ARG>(a),
        [&res](value_type const & v)
        {
            res *= v;
        },
        "prod(ARRAY_EXPRESSION)"
    );
    return res;
}

template <class ARG1, class ARG2>
enable_if_t<array_math::ArrayMathBinaryTraits<ARG1, ARG2>::value,
            bool>
operator==(ARG1 const & arg1, ARG2 const & arg2)
{
    typedef array_math::ArrayMathExpression<array_math::ArrayMathElementwiseEqual<ARG1, ARG2>> Op;

    Op op(arg1, arg2);
    if (!op.hasData())
        return op.arg1_.hasData() == op.arg2_.hasData(); // empty arrays are equal

    Shape<Op::dimension> shape(tags::size = op.ndim(), 1);
    if (!op.unifyShape(shape))
        return false;

    return all(std::move(op));
}

template <class ARG1, class ARG2>
enable_if_t<array_math::ArrayMathBinaryTraits<ARG1, ARG2>::value,
            bool>
operator!=(ARG1 const & a1, ARG2 const & a2)
{
    return !(a1 == a2);
}

} // namespace vigra

#endif // VIGRA2_ARRAY_MATH_HXX
