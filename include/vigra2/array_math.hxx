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
#include "handle_nd.hxx"
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

    // wrap a scalar HandleND<0, T> for use in array expressions
template <class T>
struct ArrayMathValueOperator
: public HandleND<0, T>
, public ArrayMathTag
{
    typedef HandleND<0, T>                       base_type;
    typedef typename base_type::difference_type  difference_type;

    difference_type shape_;

    ArrayMathValueOperator(T const & data)
    : base_type(data)
    {}

    template <class SHAPE>
    void transpose(SHAPE const & permutation) const
    {
        if(this->ndim() > 0)
            const_cast<difference_type&>(shape_) = shape_.transpose(permutation);
    }

    void setShape(difference_type shape)
    {
        shape_ = std::move(shape);
    }

    difference_type const & shape() const
    {
        return shape_;
    }

    // difference_type const & permutationToCOrder() const
    // {
        // return shape_;
    // }
};

    // wrap a an array for use in array expressions
template <class ARG>
struct ArrayMathArrayOperator
: public decltype(((ARG*)0)->handle())
, public ArrayMathTag
{
    typedef decltype(((ARG*)0)->handle())        base_type;
    typedef typename base_type::difference_type  difference_type;

    difference_type shape_, permutation_;

    ArrayMathArrayOperator(ARG const & a)
    : base_type(a.handle())
    , shape_(a.shape())
    , permutation_(array_detail::permutationToOrder(a.shape(), a.strides(), C_ORDER))
    {
        // set singleton strides to zero
        for(int k=0; k<shape_.size(); ++k)
            if(shape_[k] == 1)
                this->strides_[k] = 0;
    }

    bool noMemoryOverlap(char * p1, char * q1) const
    {
        char * p2 = (char *)this->ptr();
        char * q2 = (char *)(&(*this)[shape_ - 1] + 1);
        return q2 <= p1 || q1 <= p2;
    }

    template <class SHAPE>
    bool compatibleMemoryLayout(char * p, SHAPE const & strides) const
    {
        for(int k=0; k<strides.size(); ++k)
            if(this->strides_[k] != 0 && this->strides_[k] != strides[k])
                return false;
        return p <= (char *)this->ptr();
    }

    template <class SHAPE>
    void transpose(SHAPE const & permutation) const
    {
        const_cast<difference_type&>(this->strides_)
                                             = this->strides_.transpose(permutation);
        const_cast<difference_type&>(shape_) = shape_.transpose(permutation);
    }

    difference_type const & shape() const
    {
        return shape_;
    }

    // difference_type const & permutationToCOrder() const
    // {
        // return permutation_;
    // }
};

template <class ARG>
struct ArrayMathUnaryOperator
: public ArrayMathTag
{
    typedef typename
        std::conditional<ArrayNDConcept<ARG>::value,
                         ArrayMathArrayOperator<ARG>,
                         ARG>::type arg_type;

    typedef typename arg_type::difference_type  difference_type;
    static const int dimension                = arg_type::dimension;

    ArrayMathUnaryOperator(ARG const & a)
    : arg_(a)
    {}

    template <class SHAPE>
    ArrayIndex isConsecutive(SHAPE const &, int) const
    {
        return 0;
    }

    bool hasData() const
    {
        return arg_.hasData();
    }

    bool noMemoryOverlap(char * p1, char * q1) const
    {
        return arg_.noMemoryOverlap(p1, q1);
    }

    template <class SHAPE>
    bool compatibleMemoryLayout(char * p, SHAPE const & strides) const
    {
        return arg_.compatibleMemoryLayout(p, strides);
    }

    template <class SHAPE>
    void transpose(SHAPE const & permutation) const
    {
        arg_.transpose(permutation);
    }

    void inc() const
    {
        vigra_invariant(false, "ArrayMathUnaryOperator::inc(): internal error - should never be called.");
    }

    void move(ArrayIndex) const
    {
        vigra_invariant(false, "ArrayMathUnaryOperator::move(ArrayIndex): internal error - should never be called.");
    }

    // increment the pointer of all RHS arrays along the given 'axis'
    void inc(int axis) const
    {
        arg_.inc(axis);
    }

    // reset the pointer of all RHS arrays along the given 'axis'
    void move(int axis, ArrayIndex diff) const
    {
        arg_.move(axis, diff);
    }

    difference_type const & shape() const
    {
        return arg_.shape();
    }

    difference_type const & permutationToCOrder() const
    {
        return arg_.permutationToCOrder();
    }

    arg_type arg_;
};

#define VIGRA_ARRAY_MATH_UNARY(NAME, CALL, FUNCTION) \
template <class ARG> \
struct ArrayMath##NAME \
: public ArrayMathUnaryOperator<ARG> \
{ \
    typedef ArrayMathUnaryOperator<ARG>   base_type; \
    typedef typename base_type::arg_type  arg_type; \
    typedef decltype(CALL(**(arg_type*)0))   raw_result_type; \
    typedef typename std::conditional<std::is_same<raw_result_type, bool>::value, \
                           unsigned char, raw_result_type>::type result_type; \
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
            ArrayMath##NAME<ARG>> \
FUNCTION(ARG const & arg) \
{ \
    return ArrayMath##NAME<ARG>(arg); \
}

VIGRA_ARRAY_MATH_UNARY(Negate, -, operator-)
VIGRA_ARRAY_MATH_UNARY(Not, !, operator!)
VIGRA_ARRAY_MATH_UNARY(BitwiseNot, ~, operator~)
VIGRA_ARRAY_MATH_UNARY(Abs, vigra::abs, abs)
VIGRA_ARRAY_MATH_UNARY(Erf, vigra::erf, erf)
VIGRA_ARRAY_MATH_UNARY(Even, vigra::even, even)
VIGRA_ARRAY_MATH_UNARY(Odd, vigra::odd, odd)
VIGRA_ARRAY_MATH_UNARY(Sign, vigra::sign, sign)
VIGRA_ARRAY_MATH_UNARY(Signi, vigra::signi, signi)
VIGRA_ARRAY_MATH_UNARY(Round, vigra::round, round)
VIGRA_ARRAY_MATH_UNARY(Roundi, vigra::roundi, roundi)
VIGRA_ARRAY_MATH_UNARY(Sqrti, vigra::sqrti, sqrti)
VIGRA_ARRAY_MATH_UNARY(Sq, vigra::sq, sq)
VIGRA_ARRAY_MATH_UNARY(Norm, norm, elementwiseNorm)
VIGRA_ARRAY_MATH_UNARY(SquaredNorm, squaredNorm, elementwiseSquaredNorm)
VIGRA_ARRAY_MATH_UNARY(Sin_pi, vigra::sin_pi, sin_pi)
VIGRA_ARRAY_MATH_UNARY(Cos_pi, vigra::cos_pi, cos_pi)
VIGRA_ARRAY_MATH_UNARY(Gamma, vigra::gamma, gamma)
VIGRA_ARRAY_MATH_UNARY(Loggamma, vigra::loggamma, loggamma)
VIGRA_ARRAY_MATH_UNARY(Sqrt, std::sqrt, sqrt)
VIGRA_ARRAY_MATH_UNARY(Exp, vigra::exp, exp)
VIGRA_ARRAY_MATH_UNARY(Log, std::log, log)
VIGRA_ARRAY_MATH_UNARY(Log10, std::log10, log10)
VIGRA_ARRAY_MATH_UNARY(Sin, std::sin, sin)
VIGRA_ARRAY_MATH_UNARY(Asin, std::asin, asin)
VIGRA_ARRAY_MATH_UNARY(Cos, std::cos, cos)
VIGRA_ARRAY_MATH_UNARY(Acos, std::acos, acos)
VIGRA_ARRAY_MATH_UNARY(Tan, std::tan, tan)
VIGRA_ARRAY_MATH_UNARY(Atan, std::atan, atan)
VIGRA_ARRAY_MATH_UNARY(Floor, std::floor, floor)
VIGRA_ARRAY_MATH_UNARY(Ceil, std::ceil, ceil)
VIGRA_ARRAY_MATH_UNARY(Conj, conj, conj)
VIGRA_ARRAY_MATH_UNARY(Real, real, real)
VIGRA_ARRAY_MATH_UNARY(Imag, imag, imag)
VIGRA_ARRAY_MATH_UNARY(Arg, arg, arg)

#undef VIGRA_ARRAY_MATH_UNARY

template <class ARG>
using ArrayMathArgType =
    typename std::conditional<ArrayNDConcept<ARG>::value,
                     ArrayMathArrayOperator<ARG>,
                     typename std::conditional<ArrayMathConcept<ARG>::value,
                         ARG,
                         ArrayMathValueOperator<ARG>>::type>::type;

template <int N, int M>
struct ArrayMathUnifyShape
{
    typedef integral_minmax<N, M>      minmax;
    static const int dimension       = minmax::max > 0
                                          ? minmax::max
                                          : minmax::min;
    static const int shape_dimension = dimension == 0
                                          ? runtime_size
                                          : dimension;
    typedef Shape<shape_dimension>     shape_type;

    template <class SHAPE1, class SHAPE2>
    static shape_type exec(SHAPE1 const & s1, SHAPE2 const & s2,
                           bool throw_on_error = true)
    {
        if(s1.size() == 0)
        {
            return s2;
        }
        else if(s2.size() == 0)
        {
            return s1;
        }
        else if(s1.size() == s2.size())
        {
            shape_type res(s1.size(), DontInit);
            for(int k=0; k<s1.size(); ++k)
            {
                if(s1[k] == 1 || s1[k] == s2[k])
                {
                    res[k] = s2[k];
                }
                else if(s2[k] == 1)
                {
                    res[k] = s1[k];
                }
                else if(throw_on_error)
                {
                    std::stringstream message;
                    message << "arrayMathUnifyShape(): shape mismatch: " <<
                               s1 << " vs. " << s2 << ".";
                    vigra_precondition(false, message.str());
                }
                else
                {
                    return lemon::INVALID;
                }
            }
            return res;
        }
        else if(throw_on_error)
        {
            vigra_precondition(false,
                "arrayMathUnifyShape(): ndim mismatch.");
        }
        else
        {
            return lemon::INVALID;
        }
    }
};

template <class ARG1, class ARG2>
struct ArrayMathBinaryOperator
: public ArrayMathTag
{
    typedef ArrayMathArgType<ARG1> arg1_type;
    typedef ArrayMathArgType<ARG2> arg2_type;

    typedef ArrayMathUnifyShape<arg1_type::dimension, arg2_type::dimension> ShapeHelper;

    static const int dimension = ShapeHelper::dimension;
    typedef typename ShapeHelper::shape_type difference_type;

    arg1_type arg1_;
    arg2_type arg2_;
    difference_type shape_, permutation_;

    ArrayMathBinaryOperator(ARG1 const & a1, ARG2 const & a2)
    : arg1_(a1)
    , arg2_(a2)
    , shape_(ShapeHelper::exec(arg1_.shape(), arg2_.shape()))
    , permutation_(DontInit)
    {}

    template <class SHAPE>
    ArrayIndex isConsecutive(SHAPE const &, int) const
    {
        return 0;
    }

    bool hasData() const
    {
        return arg1_.hasData() && arg2_.hasData();
    }

    bool noMemoryOverlap(char * p1, char * q1) const
    {
        return arg1_.noMemoryOverlap(p1, q1) && arg2_.noMemoryOverlap(p1, q1);
    }

    template <class SHAPE>
    bool compatibleMemoryLayout(char * p, SHAPE const & strides) const
    {
        return arg1_.compatibleMemoryLayout(p, strides) && arg2_.compatibleMemoryLayout(p, strides);
    }

    template <class SHAPE>
    void transpose(SHAPE const & permutation) const
    {
        arg1_.transpose(permutation);
        arg2_.transpose(permutation);
        const_cast<difference_type&>(shape_) = vigra::transpose(shape_, permutation);
    }

    void inc() const
    {
        vigra_invariant(false, "ArrayMathUnaryOperator::inc(): internal error - should never be called.");
    }

    void move(ArrayIndex) const
    {
        vigra_invariant(false, "ArrayMathUnaryOperator::move(ArrayIndex): internal error - should never be called.");
    }

    // increment the pointer of all RHS arrays along the given 'axis'
    void inc(int axis) const
    {
        arg1_.inc(axis);
        arg2_.inc(axis);
    }

    // reset the pointer of all RHS arrays along the given 'axis'
    void move(int axis, ArrayIndex diff) const
    {
        arg1_.move(axis, diff);
        arg2_.move(axis, diff);
    }

    difference_type const & shape() const
    {
        return shape_;
    }

    // difference_type const & permutationToCOrder() const
    // {
        // return permutation_;
    // }
};

template <class ARG1, class ARG2,
          bool IS_ARRAY1 = ArrayNDConcept<ARG1>::value ||
                           ArrayMathConcept<ARG1>::value,
          bool IS_ARRAY2 = ArrayNDConcept<ARG2>::value ||
                           ArrayMathConcept<ARG2>::value>
struct ArrayMathBinaryTraits
{
    static const bool value = PromoteTraits<typename ARG1::value_type,
                                            typename ARG2::value_type>::value;
};


template <class ARG1, class ARG2>
struct ArrayMathBinaryTraits<ARG1, ARG2, true, false>
{
    static const bool value = PromoteTraits<typename ARG1::value_type,
                                            ARG2>::value;
};

template <class ARG1, class ARG2>
struct ArrayMathBinaryTraits<ARG1, ARG2, false, true>
{
    static const bool value = PromoteTraits<ARG1,
                                            typename ARG2::value_type>::value;
};

template <class ARG1, class ARG2>
struct ArrayMathBinaryTraits<ARG1, ARG2, false, false>
{
    static const bool value = false;
};

#define VIGRA_ARRAYMATH_BINARY_OPERATOR(NAME, CALL, FUNCTION, SEP) \
\
template <class ARG1, class ARG2> \
struct ArrayMath##NAME \
: public ArrayMathBinaryOperator<ARG1, ARG2> \
{ \
    typedef ArrayMathBinaryOperator<ARG1, ARG2>   base_type; \
    typedef typename base_type::arg1_type         arg1_type; \
    typedef typename base_type::arg2_type         arg2_type; \
    typedef decltype(CALL(**(arg1_type*)0 SEP **(arg2_type*)0)) raw_result_type; \
    typedef typename std::conditional<std::is_same<raw_result_type, bool>::value, \
                           unsigned char, raw_result_type>::type result_type; \
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
            ArrayMath##NAME<ARG1, ARG2> > \
FUNCTION(ARG1 const & a1, ARG2 const & a2) \
{ \
    return ArrayMath##NAME<ARG1, ARG2>(a1, a2); \
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

VIGRA_ARRAYMATH_BINARY_OPERATOR(Atan2, std::atan2, atan2, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Pow, vigra::pow, pow, VIGRA_COMMA)
VIGRA_ARRAYMATH_BINARY_OPERATOR(Fmod, std::fmod, fmod, VIGRA_COMMA)

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
    typedef typename std::conditional<std::is_same<raw_result_type, bool>::value, \
                           unsigned char, raw_result_type>::type result_type; \
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
            ArrayMath##NAME<ARG1, ARG2> > \
CALL(ARG1 const & a1, ARG2 const & a2) \
{ \
    return ArrayMath##NAME<ARG1, ARG2>(a1, a2); \
}

using vigra::min;
using vigra::max;
VIGRA_ARRAYMATH_MINMAX_FUNCTION(Min, min)
VIGRA_ARRAYMATH_MINMAX_FUNCTION(Max, max)

#undef VIGRA_ARRAYMATH_MINMAX_FUNCTION

template <int N>
class ArrayMathMGrid
: public ArrayMathTag
, public ShapeHandle<N>
{
public:
    typedef ShapeHandle<N>                 base_type;
    typedef Shape<N>                       shape_type;

    explicit ArrayMathMGrid(shape_type const & shape)
    : base_type(shape)
    {}

    inline void inc() const
    {
        const_cast<ArrayMathMGrid*>(this)->base_type::inc();
    }

    inline void inc(int dim) const
    {
        const_cast<ArrayMathMGrid*>(this)->base_type::inc(dim);
    }

    void move(int dim, ArrayIndex diff) const
    {
        const_cast<ArrayMathMGrid*>(this)->base_type::move(dim, diff);
    }

    template <class SHAPE>
    void transpose(SHAPE const & permutation) const
    {
        const_cast<ArrayMathMGrid*>(this)->shape_.transpose(permutation);
        const_cast<ArrayMathMGrid*>(this)->point_.transpose(permutation);
    }
};

template <int N>
ArrayMathMGrid<N>
mgrid(Shape<N> const & shape)
{
    return ArrayMathMGrid<N>(shape);
}

} // namespace array_math

using array_math::min;
using array_math::max;

template <class ARG,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline bool
all(ARG const & a)
{
    typedef typename ARG::value_type value_type;

    bool res = true;
    value_type zero = value_type();
    // FIXME: optimize memory order
    array_detail::genericArrayFunctionImpl(a, a.shape(),
        [zero, &res](value_type const & v)
        {
            if(v == zero)
                res = false;
        });
    return res;
}

template <class ARG,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline bool
any(ARG const & a)
{
    typedef typename ARG::value_type value_type;

    bool res = false;
    value_type zero = value_type();
    // FIXME: optimize memory order
    array_detail::genericArrayFunctionImpl(a, a.shape(),
        [zero, &res](value_type const & v)
        {
            if(v != zero)
                res = true;
        });
    return res;
}

template <class ARG,
          class U = PromoteType<typename ARG::value_type>,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline U
sum(ARG const & a, U res = {})
{
    typedef typename ARG::value_type value_type;
    // FIXME: optimize memory order
    array_detail::genericArrayFunctionImpl(a, a.shape(),
        [&res](value_type const & v)
        {
            res += v;
        });
    return res;
}

template <class ARG,
          class U = PromoteType<typename ARG::value_type>,
          VIGRA_REQUIRE<ArrayMathConcept<ARG>::value> >
inline U
prod(ARG const & a, U res = U{1})
{
    typedef typename ARG::value_type value_type;
    // FIXME: optimize memory order
    array_detail::genericArrayFunctionImpl(a, a.shape(),
        [&res](value_type const & v)
        {
            res *= v;
        });
    return res;
}

template <class ARG1, class ARG2>
enable_if_t<array_math::ArrayMathBinaryTraits<ARG1, ARG2>::value,
            bool>
operator==(ARG1 const & arg1, ARG2 const & arg2)
{
    typedef array_math::ArrayMathArgType<ARG1> A1;
    typedef array_math::ArrayMathArgType<ARG2> A2;
    typedef array_math::ArrayMathUnifyShape<A1::dimension, A2::dimension> ShapeHelper;
    typedef typename ShapeHelper::shape_type   shape_type;

    A1 a1(arg1);
    A2 a2(arg2);

    shape_type shape = ShapeHelper::exec(a1.shape(), a2.shape(), false);

    if(!a1.hasData() || !a2.hasData() || shape == lemon::INVALID)
        return false;

    // FIXME: optimize memory order
    // auto p  = permutationToOrder(a1.shape_, a1.strides_, C_ORDER);
    // handle.transpose(p);
    bool res = true;
    array_detail::genericArrayFunctionImpl(a1, a2, shape,
        [&res](typename A1::value_type const & u, typename A2::value_type const & v)
        {
            if(u != v)
                res = false;
        });
    return res;
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
