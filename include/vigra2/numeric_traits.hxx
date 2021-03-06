/************************************************************************/
/*                                                                      */
/*               Copyright 2014-2015 by Ullrich Koethe                  */
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

#ifndef VIGRA2_NUMERIC_TRAITS_HXX_HXX
#define VIGRA2_NUMERIC_TRAITS_HXX_HXX

#include <cmath>    // abs(double)
#include <cstdlib>  // abs(int)
#include <complex>
#include <limits>

#include "config.hxx"
#include "concepts.hxx"

namespace vigra {

namespace numeric_traits_detail {

using std::sqrt;
using concepts_detail::Unsupported;

struct MatchesAnything
{
    template <class T>
    MatchesAnything(T const &);
};

Unsupported operator+(MatchesAnything const &, MatchesAnything const &);
Unsupported operator-(MatchesAnything const &, MatchesAnything const &);
Unsupported operator*(MatchesAnything const &, MatchesAnything const &);
Unsupported operator/(MatchesAnything const &, MatchesAnything const &);
Unsupported sqrt(MatchesAnything const &);

Unsupported * check(Unsupported);

template <class T>
T * check(T const &);

template <class T1, class T2>
struct OperatorSupport
{
    typedef decltype(check(*(T1*)0 + *(T2*)0)) AddResult;
    typedef decltype(check(*(T1*)0 - *(T2*)0)) SubtractResult;
    typedef decltype(check(*(T1*)0 * *(T2*)0)) MultiplyResult;
    typedef decltype(check(*(T1*)0 / *(T2*)0)) DivideResult;
};

template <>
struct OperatorSupport<bool, bool>
{
    typedef int AddResult;
    typedef int SubtractResult;
    typedef int MultiplyResult;
    typedef int DivideResult;
};

template <class T1, class T2, class Result>
struct OperatorTraits
{
    static const bool value = true;
    using             type  = typename std::remove_pointer<Result>::type;
};

template <class T1, class T2>
struct OperatorTraits<T1, T2, Unsupported *>
{
    static const bool value = false;
};

template <class T>
struct FunctionSupport
{
    typedef decltype(check(sqrt(*(T*)0))) SqrtResult;
};

template <class T, class Result>
struct FunctionTraits
{
    static const bool value = true;
    using             type  = typename std::remove_pointer<Result>::type;
};

template <class T>
struct FunctionTraits<T, Unsupported *>
{
    static const bool value = false;
};

} // namespace numeric_traits_detail

/////////////////////////////////////////////////////////
// PromoteTraits


template <class T1, class T2 = T1>
struct AddTraits
: public numeric_traits_detail::OperatorTraits<T1, T2,
            typename numeric_traits_detail::OperatorSupport<T1, T2>::AddResult>
{};

template <class T1, class T2 = T1>
struct SubtractTraits
: public numeric_traits_detail::OperatorTraits<T1, T2,
            typename numeric_traits_detail::OperatorSupport<T1, T2>::SubtractResult>
{};

template <class T1, class T2 = T1>
struct MultiplyTraits
: public numeric_traits_detail::OperatorTraits<T1, T2,
            typename numeric_traits_detail::OperatorSupport<T1, T2>::MultiplyResult>
{};

template <class T1, class T2 = T1>
struct DivideTraits
: public numeric_traits_detail::OperatorTraits<T1, T2,
            typename numeric_traits_detail::OperatorSupport<T1, T2>::DivideResult>
{};

template <class T1, class T2 = T1>
struct PromoteTraits
: public AddTraits<T1, T2>
{};

template <class T1, class T2>
struct PromoteTraits<T1 *, T2 *>
{
    static const bool value = false;
};

template <class T>
struct PromoteTraits<T *, T *>
{
    static const bool value = true;
    using type = T *;
};

template <class T1, class T2 = T1>
using PromoteType = typename PromoteTraits<T1, T2>::type;

template <bool Cond, class T1, class T2 = T1>
using PromoteTypeIf = enable_if_t<Cond, PromoteType<T1, T2> >;

///////////////////////////////////////////////////////////////
// BoolPromote

    // replace 'bool' with 'unsigned char'
template <class T>
using BoolPromote = typename std::conditional<std::is_same<T, bool>::value, unsigned char, T>::type;

///////////////////////////////////////////////////////////////
// RealPromoteTraits

template <class T>
struct SqrtTraits
: numeric_traits_detail::FunctionTraits<T,
            typename numeric_traits_detail::FunctionSupport<T>::SqrtResult>
{};

template <class T1, class T2 = T1>
struct RealPromoteTraits
: public SqrtTraits<PromoteType<T1, T2> >
{};

template <class T1, class T2 = T1>
using RealPromoteType = typename RealPromoteTraits<T1, T2>::type;


///////////////////////////////////////////////////////////////
// NumericTraits

template<class T>
struct NumericTraits
{
    static_assert(!std::is_same<T, char>::value,
       "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

    typedef T                           Type;
    typedef PromoteType<T>              Promote;
    typedef RealPromoteType<T>          RealPromote;
    typedef Promote                     UnsignedPromote;
    typedef std::complex<RealPromote>   ComplexPromote;
    typedef T                           value_type;

    static const std::ptrdiff_t static_size = runtime_size;
};

//////////////////////////////////////////////////////
// NumericTraits specializations

template<>
struct NumericTraits<bool>
{
    typedef bool Type;
    typedef int Promote;
    typedef unsigned int UnsignedPromote;
    typedef double RealPromote;
    typedef std::complex<RealPromote> ComplexPromote;
    typedef Type value_type;

    static constexpr bool zero() noexcept { return false; }
    static constexpr bool one() noexcept { return true; }
    static constexpr bool nonZero() noexcept { return true; }
    static constexpr Type epsilon() noexcept { return true; }
    static constexpr Type smallestPositive() noexcept { return true; }
    static constexpr bool min() noexcept { return false; }
    static constexpr bool max() noexcept { return true; }

    static const bool minConst = false;
    static const bool maxConst = true;
    static const std::ptrdiff_t static_size = 1;

    static Promote toPromote(bool v) { return v ? 1 : 0; }
    static RealPromote toRealPromote(bool v) { return v ? 1.0 : 0.0; }
    static bool fromPromote(Promote v) {
        return (v == 0) ? false : true;
    }
    static bool fromRealPromote(RealPromote v) {
        return (v == 0.0) ? false : true;
    }
};

template<class T>
struct SignedNumericTraits
{
    typedef T             Type;
    typedef Type          value_type;
    typedef PromoteType<Type> Promote;
    typedef typename std::make_unsigned<Promote>::type UnsignedPromote;
    typedef RealPromoteType<Type> RealPromote;
    typedef std::complex<RealPromote> ComplexPromote;

    static constexpr Type zero() noexcept    { return 0; }
    static constexpr Type one() noexcept     { return 1; }
    static constexpr Type nonZero() noexcept { return 1; }
    static constexpr Type epsilon() noexcept { return 1; }
    static constexpr Type smallestPositive() noexcept { return 1; }
    static constexpr Type min() noexcept     { return std::numeric_limits<T>::min(); }
    static constexpr Type max() noexcept     { return std::numeric_limits<T>::max(); }

    static const Type minConst = min();
    static const Type maxConst = max();
    static const std::ptrdiff_t static_size = 1;

    static Promote      toPromote(Type v)     { return v; }
    static RealPromote  toRealPromote(Type v) { return v; }

    static Type fromPromote(Promote v)
    {
        return v <= static_cast<Promote>(min())
                   ? min()
                   : v >= static_cast<Promote>(max())
                          ? max()
                          : static_cast<Type>(v);
    }
    static Type fromRealPromote(RealPromote v)
    {
        return v <= static_cast<RealPromote>(min())
                   ? min()
                   : v >= static_cast<RealPromote>(max())
                          ? max()
                          : static_cast<Type>(std::round(v));
    }
};

template<>
struct NumericTraits<signed char> : public SignedNumericTraits<signed char> {};
template<>
struct NumericTraits<signed short> : public SignedNumericTraits<signed short> {};
template<>
struct NumericTraits<signed int> : public SignedNumericTraits<signed int> {};
template<>
struct NumericTraits<signed long> : public SignedNumericTraits<signed long> {};
template<>
struct NumericTraits<signed long long> : public SignedNumericTraits<signed long long> {};

template<class T>
struct UnsignedNumericTraits
{
    typedef T             Type;
    typedef Type          value_type;
    typedef PromoteType<Type> Promote;
    typedef typename std::make_unsigned<Promote>::type UnsignedPromote;
    typedef RealPromoteType<Type> RealPromote;
    typedef std::complex<RealPromote> ComplexPromote;

    static constexpr Type zero() noexcept    { return 0; }
    static constexpr Type one() noexcept     { return 1; }
    static constexpr Type nonZero() noexcept { return 1; }
    static constexpr Type epsilon() noexcept { return 1; }
    static constexpr Type smallestPositive() noexcept { return 1; }
    static constexpr Type min() noexcept     { return std::numeric_limits<T>::min(); }
    static constexpr Type max() noexcept     { return std::numeric_limits<T>::max(); }

    static const Type minConst = min();
    static const Type maxConst = max();
    static const std::ptrdiff_t static_size = 1;

    static Promote      toPromote(Type v)     { return v; }
    static RealPromote  toRealPromote(Type v) { return v; }

    static Type fromPromote(Promote v)
    {
        return v <= static_cast<Promote>(zero())
                   ? zero()
                   : v >= static_cast<Promote>(max())
                          ? max()
                          : static_cast<Type>(v);
    }
    static Type fromRealPromote(RealPromote v)
    {
        return v <= static_cast<RealPromote>(zero())
                   ? zero()
                   : v >= static_cast<RealPromote>(max())
                          ? max()
                          : static_cast<Type>(std::round(v));
    }
};

template<>
struct NumericTraits<unsigned char> : public UnsignedNumericTraits<unsigned char> {};
template<>
struct NumericTraits<unsigned short> : public UnsignedNumericTraits<unsigned short> {};
template<>
struct NumericTraits<unsigned int> : public UnsignedNumericTraits<unsigned int> {};
template<>
struct NumericTraits<unsigned long> : public UnsignedNumericTraits<unsigned long> {};
template<>
struct NumericTraits<unsigned long long> : public UnsignedNumericTraits<unsigned long long> {};

template<class T>
struct FloatNumericTraits
{
    typedef T    Type;
    typedef Type value_type;
    typedef Type Promote;
    typedef Type UnsignedPromote;
    typedef Type RealPromote;
    typedef std::complex<RealPromote> ComplexPromote;

    static constexpr Type zero() noexcept { return 0.0; }
    static constexpr Type one() noexcept { return 1.0; }
    static constexpr Type nonZero() noexcept { return 1.0; }
    static constexpr Type epsilon() noexcept { return std::numeric_limits<Type>::epsilon(); }
    static constexpr Type smallestPositive() noexcept { return std::numeric_limits<Type>::min(); }
    static constexpr Type min() noexcept { return std::numeric_limits<Type>::lowest(); }
    static constexpr Type max() noexcept { return std::numeric_limits<Type>::max(); }

    static const std::ptrdiff_t static_size = 1;

    static Promote      toPromote(Type v) { return v; }
    static RealPromote  toRealPromote(Type v) { return v; }
    static Type         fromPromote(Promote v) { return v; }
    static Type         fromRealPromote(RealPromote v)
    {
        return v <= static_cast<RealPromote>(min())
                   ? min()
                   : v >= static_cast<RealPromote>(max())
                          ? max()
                          : static_cast<Type>(v);
    }
};

template<>
struct NumericTraits<float> : public FloatNumericTraits<float> {};
template<>
struct NumericTraits<double> : public FloatNumericTraits<double> {};
template<>
struct NumericTraits<long double> : public FloatNumericTraits<long double> {};

template<class T>
struct NumericTraits<std::complex<T> >
{
    typedef std::complex<T> Type;
    typedef std::complex<typename NumericTraits<T>::Promote> Promote;
    typedef std::complex<typename NumericTraits<T>::UnsignedPromote> UnsignedPromote;
    typedef std::complex<typename NumericTraits<T>::RealPromote> RealPromote;
    typedef std::complex<RealPromote> ComplexPromote;
    typedef T value_type;

    static Type zero() { return Type(0.0); }
    static Type one() { return Type(1.0); }
    static Type nonZero() { return one(); }
    static Type epsilon() { return Type(NumericTraits<T>::epsilon()); }
    static Type smallestPositive() { return Type(NumericTraits<T>::smallestPositive()); }

    static const std::ptrdiff_t static_size = 2;

    static Promote toPromote(Type const & v) { return v; }
    static Type    fromPromote(Promote const & v) { return v; }
    static Type    fromRealPromote(RealPromote v) { return Type(v); }
};

///////////////////////////////////////////////////////////////
// UninitializedMemoryTraits

template<class T>
struct UninitializedMemoryTraits
{
    static const bool value = std::is_scalar<T>::value || std::is_pod<T>::value;
};


///////////////////////////////////////////////////////////////
// NormTraits

template<class T>
struct NormTraits;

namespace detail {

template <class T, bool scalar = std::is_arithmetic<T>::value>
struct NormOfScalarImpl;

template <class T>
struct NormOfScalarImpl<T, false>
{
    static const bool value = false;
    typedef void *            NormType;
    typedef void *            SquaredNormType;
};

template <class T>
struct NormOfScalarImpl<T, true>
{
    static const bool value = true;
    typedef T                 NormType;
    typedef typename
        std::conditional<sizeof(T) == 4 && std::is_integral<T>::value,
                unsigned long long,
                typename NumericTraits<T>::UnsignedPromote>::type
        SquaredNormType;
};

template <class T, bool integral = std::is_integral<T>::value,
                   bool floating = std::is_floating_point<T>::value>
struct NormOfArrayElementsImpl;

template <>
struct NormOfArrayElementsImpl<void *, false, false>
{
    typedef void *  NormType;
    typedef void *  SquaredNormType;
};

template <class T>
struct NormOfArrayElementsImpl<T, false, false>
{
    typedef typename NormTraits<T>::NormType         NormType;
    typedef typename NormTraits<T>::SquaredNormType  SquaredNormType;
};

template <class T>
struct NormOfArrayElementsImpl<T, true, false>
{
    static_assert(!std::is_same<T, char>::value,
       "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

    typedef double              NormType;
    typedef unsigned long long  SquaredNormType;
};

template <class T>
struct NormOfArrayElementsImpl<T, false, true>
{
    typedef double              NormType;
    typedef double              SquaredNormType;
};

template <>
struct NormOfArrayElementsImpl<long double, false, true>
{
    typedef long double         NormType;
    typedef long double         SquaredNormType;
};

template <class ARRAY>
struct NormOfVectorImpl
{
    static void * test(...);

    template <class U>
    static typename U::value_type test(U*, typename U::value_type * = 0);

    typedef decltype(test((ARRAY*)0)) T;

    static const bool value = !std::is_same<T, void*>::value;

    typedef typename NormOfArrayElementsImpl<T>::NormType         NormType;
    typedef typename NormOfArrayElementsImpl<T>::SquaredNormType  SquaredNormType;
};


} // namespace detail

    /* NormTraits<T> implement the following default rules, which are
       designed to minimize the possibility of overflow:
        * T is a 32-bit integer type:
               NormType is T itself,
               SquaredNormType is 'unsigned long long'
        * T is another built-in arithmetic type:
               NormType is T itself,
               SquaredNormType is the NumericTraits<T>::UnsignedPromote
        * T is a container of 'long double':
               NormType and SquaredNormType are 'long double'
        * T is a container of another built-in arithmetic type:
               NormType is 'double',
               SquaredNormType is 'unsigned long long'
        * T is a container of some other type:
               NormType is the element's norm type,
               SquaredNormType is the element's squared norm type
       Containers are recognized by having an embedded typedef 'value_type'.

       To change the behavior for a particular case or extend it to cases
       not covered here, simply specialize the NormTraits template.
    */
template<class T>
struct NormTraits
{
    static_assert(!std::is_same<T, char>::value,
       "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

    typedef detail::NormOfScalarImpl<T>   NormOfScalar;
    typedef detail::NormOfVectorImpl<T>   NormOfVector;

    static const bool value = NormOfScalar::value || NormOfVector::value;

    static_assert(value, "NormTraits<T> are undefined for type T.");

    typedef typename std::conditional<NormOfVector::value,
                typename NormOfVector::NormType,
                typename NormOfScalar::NormType>::type
            NormType;

    typedef typename std::conditional<NormOfVector::value,
                typename NormOfVector::SquaredNormType,
                typename NormOfScalar::SquaredNormType>::type
            SquaredNormType;
};

template <class T>
using SquaredNormType = typename NormTraits<T>::SquaredNormType;

template <class T>
using NormType = typename NormTraits<T>::NormType;

///////////////////////////////////////////////////////////////
// ValueTypeTraits

template <class CONTAINER>
struct ValueTypeTraits
{
    typedef typename std::remove_reference<CONTAINER>::type C;

    static concepts_detail::Unsupported * test(...);

    template <class U>
    static typename U::value_type * test(U const *, typename U::value_type * = 0);

    typedef typename std::remove_pointer<decltype(test((C*)0))>::type TestResult;

    static const bool value = !std::is_same<concepts_detail::Unsupported, TestResult>::value;
    typedef typename
        std::conditional<value,
                         typename std::conditional<std::is_const<C>::value,
                                                   typename std::add_const<TestResult>::type,
                                                   TestResult>::type,
                         void>::type
        type;
};

///////////////////////////////////////////////////////////////
// RequiresExplicitCast

namespace detail {

template <class T>
struct RequiresExplicitCast {
    template <class U>
    static U const & cast(U const & v)
        { return v; }
};

#define VIGRA_SPECIALIZED_CAST(type) \
template <> \
struct RequiresExplicitCast<type> { \
    static type cast(float v) \
        { return NumericTraits<type>::fromRealPromote(v); } \
    static type cast(double v) \
        { return NumericTraits<type>::fromRealPromote(v); } \
    static type cast(type v) \
        { return v; } \
    template <class U> \
    static type cast(U v) \
        { return static_cast<type>(v); } \
\
};

VIGRA_SPECIALIZED_CAST(signed char)
VIGRA_SPECIALIZED_CAST(unsigned char)
VIGRA_SPECIALIZED_CAST(short)
VIGRA_SPECIALIZED_CAST(unsigned short)
VIGRA_SPECIALIZED_CAST(int)
VIGRA_SPECIALIZED_CAST(unsigned int)
VIGRA_SPECIALIZED_CAST(long)
VIGRA_SPECIALIZED_CAST(unsigned long)

#undef VIGRA_SPECIALIZED_CAST

template <>
struct RequiresExplicitCast<bool> {
    template <class U>
    static bool cast(U v)
    { return v == NumericTraits<U>::zero()
                ? false
                : true; }
};

template <>
struct RequiresExplicitCast<float> {
    static float cast(int v)
        { return (float)v; }

    static float cast(unsigned int v)
        { return (float)v; }

    static float cast(long v)
        { return (float)v; }

    static float cast(unsigned long v)
        { return (float)v; }

    static float cast(long long v)
        { return (float)v; }

    static float cast(unsigned long long v)
        { return (float)v; }

    static float cast(double v)
        { return (float)v; }

    static float cast(long double v)
        { return (float)v; }

    template <class U>
    static U cast(U v)
        { return v; }
};

template <>
struct RequiresExplicitCast<double> {
    static double cast(long long v)
        { return (double)v; }

    static double cast(unsigned long long v)
        { return (double)v; }

    template <class U>
    static U cast(U v)
        { return v; }
};

} // namespace detail

// comparison with tolerance

namespace numeric_traits_detail {

// both f1 and f2 are unsigned here
template<class FPT>
inline
FPT safeFloatDivision( FPT f1, FPT f2 )
{
    return  f2 < NumericTraits<FPT>::one() && f1 > f2 * NumericTraits<FPT>::max()
                ? NumericTraits<FPT>::max()
                : (f2 > NumericTraits<FPT>::one() && f1 < f2 * NumericTraits<FPT>::smallestPositive()) ||
                   f1 == NumericTraits<FPT>::zero()
                     ? NumericTraits<FPT>::zero()
                     : f1/f2;
}

} // namespace numeric_traits_detail

    /** \brief Tolerance based floating-point equality.

        Check whether two floating point numbers are equal within the given tolerance.
        This is useful because floating point numbers that should be equal in theory are
        rarely exactly equal in practice. If the tolerance \a epsilon is not given,
        twice the machine epsilon is used.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2>
typename enable_if<std::is_floating_point<PromoteType<T1, T2> >::value,
                        bool>::type
closeAtTolerance(T1 l, T2 r, PromoteType<T1, T2> epsilon)
{
    using std::abs;
    typedef PromoteType<T1, T2> T;
    if(l == 0.0)
        return abs(r) <= epsilon;
    if(r == 0.0)
        return abs(l) <= epsilon;
    T diff = abs( l - r );
    T d1   = numeric_traits_detail::safeFloatDivision<T>( diff, abs( r ) );
    T d2   = numeric_traits_detail::safeFloatDivision<T>( diff, abs( l ) );

    return (d1 <= epsilon && d2 <= epsilon);
}

template <class T1, class T2>
inline
typename enable_if<std::is_floating_point<PromoteType<T1, T2> >::value,
                        bool>::type
closeAtTolerance(T1 l, T2 r)
{
    typedef PromoteType<T1, T2> T;
    return closeAtTolerance(l, r, T(2.0) * NumericTraits<T>::epsilon());
}

    /** \brief Tolerance based floating-point less-or-equal.

        Check whether two floating point numbers are less or equal within the given tolerance.
        That is, \a l can actually be greater than \a r within the given \a epsilon.
        This is useful because floating point numbers that should be equal in theory are
        rarely exactly equal in practice. If the tolerance \a epsilon is not given,
        twice the machine epsilon is used.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2>
inline
typename enable_if<std::is_floating_point<PromoteType<T1, T2> >::value,
                        bool>::type
lessEqualAtTolerance(T1 l, T2 r, PromoteType<T1, T2> epsilon)
{
    return l < r || closeAtTolerance(l, r, epsilon);
}

template <class T1, class T2>
inline
typename enable_if<std::is_floating_point<PromoteType<T1, T2> >::value,
                        bool>::type
lessEqualAtTolerance(T1 l, T2 r)
{
    typedef PromoteType<T1, T2> T;
    return lessEqualAtTolerance(l, r, T(2.0) * NumericTraits<T>::epsilon());
}

    /** \brief Tolerance based floating-point greater-or-equal.

        Check whether two floating point numbers are greater or equal within the given tolerance.
        That is, \a l can actually be less than \a r within the given \a epsilon.
        This is useful because floating point numbers that should be equal in theory are
        rarely exactly equal in practice. If the tolerance \a epsilon is not given,
        twice the machine epsilon is used.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2>
inline
typename enable_if<std::is_floating_point<PromoteType<T1, T2> >::value,
                        bool>::type
greaterEqualAtTolerance(T1 l, T2 r, PromoteType<T1, T2> epsilon)
{
    return r < l || closeAtTolerance(l, r, epsilon);
}

template <class T1, class T2>
inline bool greaterEqualAtTolerance(T1 l, T2 r)
{
    typedef PromoteType<T1, T2> T;
    return greaterEqualAtTolerance(l, r, T(2.0) * NumericTraits<T>::epsilon());
}

} // namespace vigra

#endif // VIGRA2_NUMERIC_TRAITS_HXX_HXX
