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

#ifndef VIGRA2_MATHUTIL_HXX
#define VIGRA2_MATHUTIL_HXX

#ifdef _MSC_VER
# pragma warning (disable: 4996) // hypot/_hypot confusion
#endif

#include <cmath>
#include <cstdlib>
#include <complex>
#include <algorithm>
#include "config.hxx"
#include "error.hxx"
#include "numeric_traits.hxx"
#include "sized_int.hxx"
#include "concepts.hxx"

/** \page MathConstants Mathematical Constants

    <TT>M_PI, M_SQRT2 etc.</TT>

    <b>\#include</b> \<vigra2/mathutil.hxx\>

    Since mathematical constants such as <TT>M_PI</TT> and <TT>M_SQRT2</TT>
    are not officially standardized, we provide definitions here for those
    compilers that don't support them.

    \code
    #ifndef M_PI
    #    define M_PI     3.14159265358979323846
    #endif

    #ifndef M_SQRT2
    #    define M_2_PI   0.63661977236758134308
    #endif

    #ifndef M_PI_2
    #    define M_PI_2   1.57079632679489661923
    #endif

    #ifndef M_PI_4
    #    define M_PI_4   0.78539816339744830962
    #endif

    #ifndef M_SQRT2
    #    define M_SQRT2  1.41421356237309504880
    #endif

    #ifndef M_EULER_GAMMA
    #    define M_EULER_GAMMA  0.5772156649015329
    #endif
    \endcode
*/
#ifndef M_PI
#    define M_PI     3.14159265358979323846
#endif

#ifndef M_2_PI
#    define M_2_PI   0.63661977236758134308
#endif

#ifndef M_PI_2
#    define M_PI_2   1.57079632679489661923
#endif

#ifndef M_PI_4
#    define M_PI_4   0.78539816339744830962
#endif

#ifndef M_SQRT2
#    define M_SQRT2  1.41421356237309504880
#endif

#ifndef M_E
#    define M_E      2.71828182845904523536
#endif

#ifndef M_EULER_GAMMA
#    define M_EULER_GAMMA  0.5772156649015329
#endif

namespace vigra {

/** \addtogroup MathFunctions Mathematical Functions

    Useful mathematical functions and functors.
*/
//@{

// import abs(float), abs(double), abs(long double) from <cmath>
//        abs(int), abs(long), abs(long long) from <cstdlib>
//        abs(std::complex<T>) from <complex>
using std::abs;
using std::fabs;

// import <cmath>
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;

using std::cosh;
using std::sinh;
using std::tanh;
using std::acosh;
using std::asinh;
using std::atanh;

using std::sqrt;
using std::cbrt;

using std::exp;
using std::exp2;
using std::expm1;
using std::log;
using std::log2;
using std::log10;
using std::log1p;
using std::logb;
using std::ilogb;

using std::ceil;
using std::floor;
using std::trunc;
using std::round;
using std::lround;
using std::llround;

using std::erf;
using std::erfc;
using std::tgamma;
using std::lgamma;

using std::isfinite;
using std::isinf;
using std::isnan;
using std::isnormal;
using std::signbit;

using std::atan2;
using std::copysign;
using std::fdim;
using std::fmax;
using std::fmin;
using std::fmod;
using std::hypot;
using std::pow;

/**********************************************************/
/*                                                        */
/*                          sq()                          */
/*                                                        */
/**********************************************************/

    /** \brief The square function.

        <tt>sq(x) = x*x</tt> is needed so often that it makes sense to define it as a function.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline PromoteType<T>
sq(T t)
{
    return t*t;
}

/**********************************************************/
/*                                                        */
/*                         min()                          */
/*                                                        */
/**********************************************************/

    /** \brief A proper minimum function.

        The <tt>std::min</tt> template matches everything -- this is way too
        greedy to be useful. VIGRA implements the basic <tt>min</tt> function
        only for arithmetic types and provides explicit overloads for everything
        else. Moreover, VIGRA's <tt>min</tt> function also computes the minimum
        between two different types, as long as they have a <tt>std::common_type</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2,
          VIGRA_REQUIRE<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value> >
inline common_type_t<T1, T2>
min(T1 const & t1, T2 const & t2)
{
    return std::min<common_type_t<T1, T2>>(t1, t2);
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T const &
min(T const & t1, T const & t2)
{
    return std::min(t1, t2);
}

/**********************************************************/
/*                                                        */
/*                         max()                          */
/*                                                        */
/**********************************************************/

    /** \brief A proper maximum function.

        The <tt>std::max</tt> template matches everything -- this is way too
        greedy to be useful. VIGRA implements the basic <tt>max</tt> function
        only for arithmetic types and provides explicit overloads for everything
        else. Moreover, VIGRA's <tt>max</tt> function also computes the maximum
        between two different types, as long as they have a <tt>std::common_type</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2,
          VIGRA_REQUIRE<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value> >
inline common_type_t<T1, T2>
max(T1 const & t1, T2 const & t2)
{
    return std::max<typename common_type<T1, T2>::type>(t1, t2);
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T const &
max(T const & t1, T const & t2)
{
    return std::max(t1, t2);
}

/**********************************************************/
/*                                                        */
/*                   floor(), ceil()                      */
/*                                                        */
/**********************************************************/

    // add missing floor() and ceil() overloads for integral types

#define VIGRA_DEFINE_INTEGER_FLOOR_CEIL(T) \
    inline T floor(signed T t) { return t; } \
    inline T floor(unsigned T t) { return t; } \
    inline T ceil(signed T t) { return t; } \
    inline T ceil(unsigned T t) { return t; }

VIGRA_DEFINE_INTEGER_FLOOR_CEIL(char)
VIGRA_DEFINE_INTEGER_FLOOR_CEIL(short)
VIGRA_DEFINE_INTEGER_FLOOR_CEIL(int)
VIGRA_DEFINE_INTEGER_FLOOR_CEIL(long)
VIGRA_DEFINE_INTEGER_FLOOR_CEIL(long long)

#undef VIGRA_DEFINE_INTEGER_FLOOR_CEIL

/**********************************************************/
/*                                                        */
/*                       isfinite()                       */
/*                                                        */
/**********************************************************/

    // add missing isfinite() overloads for integral types

#define VIGRA_DEFINE_INTEGER_ISFINITE(T) \
    inline bool isfinite(signed T t) { return true; } \
    inline bool isfinite(unsigned T t) { return true; }

VIGRA_DEFINE_INTEGER_ISFINITE(char)
VIGRA_DEFINE_INTEGER_ISFINITE(short)
VIGRA_DEFINE_INTEGER_ISFINITE(int)
VIGRA_DEFINE_INTEGER_ISFINITE(long)
VIGRA_DEFINE_INTEGER_ISFINITE(long long)

#undef VIGRA_DEFINE_INTEGER_ISFINITE

/**********************************************************/
/*                                                        */
/*                         abs()                          */
/*                                                        */
/**********************************************************/

// define the missing variants of abs() to avoid 'ambiguous overload'
// errors in template functions
#define VIGRA_DEFINE_UNSIGNED_ABS(T) \
    inline T abs(T t) { return t; }

VIGRA_DEFINE_UNSIGNED_ABS(bool)
VIGRA_DEFINE_UNSIGNED_ABS(unsigned char)
VIGRA_DEFINE_UNSIGNED_ABS(unsigned short)
VIGRA_DEFINE_UNSIGNED_ABS(unsigned int)
VIGRA_DEFINE_UNSIGNED_ABS(unsigned long)
VIGRA_DEFINE_UNSIGNED_ABS(unsigned long long)

#undef VIGRA_DEFINE_UNSIGNED_ABS

#define VIGRA_DEFINE_MISSING_ABS(T) \
    inline T abs(T t) { return t < 0 ? static_cast<T>(-t) : t; }

VIGRA_DEFINE_MISSING_ABS(signed char)
VIGRA_DEFINE_MISSING_ABS(signed short)

#if defined(_MSC_VER) && _MSC_VER < 1600
VIGRA_DEFINE_MISSING_ABS(signed long long)
#endif

#undef VIGRA_DEFINE_MISSING_ABS

/**********************************************************/
/*                                                        */
/*                     squaredNorm()                      */
/*                                                        */
/**********************************************************/

template <class T>
inline SquaredNormType<std::complex<T> >
squaredNorm(std::complex<T> const & t)
{
    return sq(t.real()) + sq(t.imag());
}

#ifdef DOXYGEN // only for documentation
    /** \brief The squared norm of a numerical object.

        <ul>
        <li>For scalar types: equals <tt>vigra::sq(t)</tt>.
        <li>For vectorial types (including TinyArray): equals <tt>vigra::dot(t, t)</tt>.
        <li>For complex number types: equals <tt>vigra::sq(t.real()) + vigra::sq(t.imag())</tt>.
        <li>For array and matrix types: results in the squared Frobenius norm (sum of squares of the matrix elements).
        </ul>
    */
SquaredNormType<T> squaredNorm(T const & t);

#endif

/**********************************************************/
/*                                                        */
/*                          norm()                        */
/*                                                        */
/**********************************************************/

    /** \brief The norm of a numerical object.

        For scalar types: implemented as <tt>abs(t)</tt><br>
        otherwise: implemented as <tt>sqrt(squaredNorm(t))</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T>
inline auto
norm(T const & t) -> decltype(sqrt(squaredNorm(t)))
{
    return sqrt(squaredNorm(t));
}

#define VIGRA_DEFINE_NORM(T) \
    inline SquaredNormType<T> squaredNorm(T t) { return sq(t); } \
    inline NormType<T> norm(T t) { return abs(t); } \
    inline SquaredNormType<T> sizeDividedSquaredNorm(T t) { return sq(t); } \
    inline NormType<T> sizeDividedNorm(T t) { return abs(t); }

VIGRA_DEFINE_NORM(signed char)
VIGRA_DEFINE_NORM(unsigned char)
VIGRA_DEFINE_NORM(short)
VIGRA_DEFINE_NORM(unsigned short)
VIGRA_DEFINE_NORM(int)
VIGRA_DEFINE_NORM(unsigned int)
VIGRA_DEFINE_NORM(long)
VIGRA_DEFINE_NORM(unsigned long)
VIGRA_DEFINE_NORM(long long)
VIGRA_DEFINE_NORM(unsigned long long)
VIGRA_DEFINE_NORM(float)
VIGRA_DEFINE_NORM(double)
VIGRA_DEFINE_NORM(long double)

#undef VIGRA_DEFINE_NORM

/**********************************************************/
/*                                                        */
/*                           dot()                        */
/*                                                        */
/**********************************************************/

// scalar dot is needed for generic functions that should work with
// scalars and vectors alike

#define VIGRA_DEFINE_SCALAR_DOT(T) \
    inline PromoteType<T> dot(T l, T r) { return l*r; }

VIGRA_DEFINE_SCALAR_DOT(unsigned char)
VIGRA_DEFINE_SCALAR_DOT(unsigned short)
VIGRA_DEFINE_SCALAR_DOT(unsigned int)
VIGRA_DEFINE_SCALAR_DOT(unsigned long)
VIGRA_DEFINE_SCALAR_DOT(unsigned long long)
VIGRA_DEFINE_SCALAR_DOT(signed char)
VIGRA_DEFINE_SCALAR_DOT(signed short)
VIGRA_DEFINE_SCALAR_DOT(signed int)
VIGRA_DEFINE_SCALAR_DOT(signed long)
VIGRA_DEFINE_SCALAR_DOT(signed long long)
VIGRA_DEFINE_SCALAR_DOT(float)
VIGRA_DEFINE_SCALAR_DOT(double)
VIGRA_DEFINE_SCALAR_DOT(long double)

#undef VIGRA_DEFINE_SCALAR_DOT

/**********************************************************/
/*                                                        */
/*                           pow()                        */
/*                                                        */
/**********************************************************/

// support 'double' exponents for all floating point versions of pow()
inline float pow(float v, double e)
{
    return std::pow(v, (float)e);
}

inline long double pow(long double v, double e)
{
    return std::pow(v, (long double)e);
}

/**********************************************************/
/*                                                        */
/*                         roundi()                       */
/*                                                        */
/**********************************************************/

    // round and cast to nearest long int
inline long int roundi(double t)
{
     return lround(t);
}

/**********************************************************/
/*                                                        */
/*                       ceilPower2()                     */
/*                                                        */
/**********************************************************/

    /** \brief Round up to the nearest power of 2.

        Efficient algorithm for finding the smallest power of 2 which is not smaller than \a x
        (function clp2() from Henry Warren: "Hacker's Delight", Addison-Wesley, 2003,
         see http://www.hackersdelight.org/).
        If \a x > 2^31, the function will return 0 because integer arithmetic is defined modulo 2^32.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline uint32_t ceilPower2(uint32_t x)
{
    if(x == 0) return 0;

    x = x - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >>16);
    return x + 1;
}

/**********************************************************/
/*                                                        */
/*                       floorPower2()                    */
/*                                                        */
/**********************************************************/

    /** \brief Round down to the nearest power of 2.

        Efficient algorithm for finding the largest power of 2 which is not greater than \a x
        (function flp2() from Henry Warren: "Hacker's Delight", Addison-Wesley, 2003,
         see http://www.hackersdelight.org/).

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline uint32_t floorPower2(uint32_t x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >>16);
    return x - (x >> 1);
}

/**********************************************************/
/*                                                        */
/*                          log2i()                       */
/*                                                        */
/**********************************************************/

namespace detail {

template <class T>
struct IntLog2
{
    static int32_t table[64];
};

template <class T>
int32_t IntLog2<T>::table[64] = {
         -1,  0,  -1,  15,  -1,  1,  28,  -1,  16,  -1,  -1,  -1,  2,  21,
         29,  -1,  -1,  -1,  19,  17,  10,  -1,  12,  -1,  -1,  3,  -1,  6,
         -1,  22,  30,  -1,  14,  -1,  27,  -1,  -1,  -1,  20,  -1,  18,  9,
         11,  -1,  5,  -1,  -1,  13,  26,  -1,  -1,  8,  -1,  4,  -1,  25,
         -1,  7,  24,  -1,  23,  -1,  31,  -1};

} // namespace detail

    /** \brief Compute the base-2 logarithm of an integer.

        Returns the position of the left-most 1-bit in the given number \a x, or
        -1 if \a x == 0. That is,

        \code
        assert(k >= 0 && k < 32 && log2i(1 << k) == k);
        \endcode

        The function uses Robert Harley's algorithm to determine the number of leading zeros
        in \a x (algorithm nlz10() at http://www.hackersdelight.org/). But note that the functions
        \ref floorPower2() or \ref ceilPower2() are more efficient and should be preferred when possible.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline int32_t log2i(uint32_t x)
{
    // Propagate leftmost 1-bit to the right.
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >>16);
    x = x*0x06EB14F9; // Multiplier is 7*255**3.
    return detail::IntLog2<int32_t>::table[x >> 26];
}

/**********************************************************/
/*                                                        */
/*                         power()                        */
/*                                                        */
/**********************************************************/

namespace detail {

template <class V, unsigned>
struct cond_mult
{
    static V call(const V & x, const V & y) { return x * y; }
};
template <class V>
struct cond_mult<V, 0>
{
    static V call(const V &, const V & y) { return y; }
};

template <class V, unsigned n>
struct power_static
{
    static V call(const V & x)
    {
        return n / 2
            ? cond_mult<V, n & 1>::call(x, power_static<V, n / 2>::call(x * x))
            : n & 1 ? x : V();
    }
};

template <class V>
struct power_static<V, 0>
{
    static V call(const V & /* x */)
    {
        return V(1);
    }
};

} // namespace detail

    /** \brief Exponentiation to a positive integer power by squaring.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <unsigned n, class V>
inline V power(const V & x)
{
    return detail::power_static<V, n>::call(x);
}

/**********************************************************/
/*                                                        */
/*                          sqrti()                       */
/*                                                        */
/**********************************************************/

namespace detail {

template <class T>
struct IntSquareRoot
{
    static uint32_t sqq_table[];
    static uint32_t exec(uint32_t v);
};

template <class T>
uint32_t IntSquareRoot<T>::sqq_table[] = {
           0,  16,  22,  27,  32,  35,  39,  42,  45,  48,  50,  53,  55,  57,
          59,  61,  64,  65,  67,  69,  71,  73,  75,  76,  78,  80,  81,  83,
          84,  86,  87,  89,  90,  91,  93,  94,  96,  97,  98,  99, 101, 102,
         103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118,
         119, 120, 121, 122, 123, 124, 125, 126, 128, 128, 129, 130, 131, 132,
         133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 144, 145,
         146, 147, 148, 149, 150, 150, 151, 152, 153, 154, 155, 155, 156, 157,
         158, 159, 160, 160, 161, 162, 163, 163, 164, 165, 166, 167, 167, 168,
         169, 170, 170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 178,
         179, 180, 181, 181, 182, 183, 183, 184, 185, 185, 186, 187, 187, 188,
         189, 189, 190, 191, 192, 192, 193, 193, 194, 195, 195, 196, 197, 197,
         198, 199, 199, 200, 201, 201, 202, 203, 203, 204, 204, 205, 206, 206,
         207, 208, 208, 209, 209, 210, 211, 211, 212, 212, 213, 214, 214, 215,
         215, 216, 217, 217, 218, 218, 219, 219, 220, 221, 221, 222, 222, 223,
         224, 224, 225, 225, 226, 226, 227, 227, 228, 229, 229, 230, 230, 231,
         231, 232, 232, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238,
         239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246,
         246, 247, 247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253,
         253, 254, 254, 255
};

template <class T>
uint32_t IntSquareRoot<T>::exec(uint32_t x)
{
    uint32_t xn;
    if (x >= 0x10000)
        if (x >= 0x1000000)
            if (x >= 0x10000000)
                if (x >= 0x40000000) {
                    if (x >= (uint32_t)65535*(uint32_t)65535)
                        return 65535;
                    xn = sqq_table[x>>24] << 8;
                } else
                    xn = sqq_table[x>>22] << 7;
            else
                if (x >= 0x4000000)
                    xn = sqq_table[x>>20] << 6;
                else
                    xn = sqq_table[x>>18] << 5;
        else {
            if (x >= 0x100000)
                if (x >= 0x400000)
                    xn = sqq_table[x>>16] << 4;
                else
                    xn = sqq_table[x>>14] << 3;
            else
                if (x >= 0x40000)
                    xn = sqq_table[x>>12] << 2;
                else
                    xn = sqq_table[x>>10] << 1;

            goto nr1;
        }
    else
        if (x >= 0x100) {
            if (x >= 0x1000)
                if (x >= 0x4000)
                    xn = (sqq_table[x>>8] >> 0) + 1;
                else
                    xn = (sqq_table[x>>6] >> 1) + 1;
            else
                if (x >= 0x400)
                    xn = (sqq_table[x>>4] >> 2) + 1;
                else
                    xn = (sqq_table[x>>2] >> 3) + 1;

            goto adj;
        } else
            return sqq_table[x] >> 4;

    /* Run two iterations of the standard convergence formula */

    xn = (xn + 1 + x / xn) / 2;
  nr1:
    xn = (xn + 1 + x / xn) / 2;
  adj:

    if (xn * xn > x) /* Correct rounding if necessary */
        xn--;

    return xn;
}

} // namespace detail

    /** \brief Signed integer square root.

        Useful for fast fixed-point computations.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline int32_t sqrti(int32_t v)
{
    if(v < 0)
        throw std::domain_error("sqrti(int32_t): negative argument.");
    return (int32_t)detail::IntSquareRoot<uint32_t>::exec((uint32_t)v);
}

    /** \brief Unsigned integer square root.

        Useful for fast fixed-point computations.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline uint32_t sqrti(uint32_t v)
{
    return detail::IntSquareRoot<uint32_t>::exec(v);
}

/**********************************************************/
/*                                                        */
/*                    sign(), signi()                     */
/*                                                        */
/**********************************************************/

    /** \brief The sign function.

        Returns 1, 0, or -1 depending on the sign of \a t, but with the same type as \a t.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T
sign(T t)
{
    return t > T()
               ? T(1)
               : t < T()
                    ? T(-1)
                    : T();
}

    /** \brief The integer sign function.

        Returns 1, 0, or -1 depending on the sign of \a t, converted to int.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline int
signi(T t)
{
    return t > T()
               ? 1
               : t < T()
                    ? -1
                    : 0;
}

    // transfer sign of t2 to t1
template <class T1, class T2>
inline T1 sign(T1 t1, T2 t2)
{
    return copysign(t1, t2);
}


/**********************************************************/
/*                                                        */
/*                       even(), odd()                    */
/*                                                        */
/**********************************************************/

template <class T,
          VIGRA_REQUIRE<std::is_integral<T>::value> >
inline bool
even(T const t)
{
    return (t&1) == 0;
}

template <class T,
          VIGRA_REQUIRE<std::is_integral<T>::value> >
inline bool
odd(T const t)
{
    return (t&1) != 0;
}

/**********************************************************/
/*                                                        */
/*                   sin_pi(), cos_pi()                   */
/*                                                        */
/**********************************************************/

    /** \brief sin(pi*x).

        Essentially calls <tt>std::sin(M_PI*x)</tt> but uses a more accurate implementation
        to make sure that <tt>sin_pi(1.0) == 0.0</tt> (which does not hold for
        <tt>std::sin(M_PI)</tt> due to round-off error), and <tt>sin_pi(0.5) == 1.0</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class REAL,
          VIGRA_REQUIRE<std::is_floating_point<REAL>::value> >
REAL sin_pi(REAL x)
{
    if(x < 0.0)
        return -sin_pi(-x);
    if(x < 0.5)
        return std::sin(M_PI * x);

    bool invert = false;
    if(x < 1.0)
    {
        invert = true;
        x = -x;
    }

    REAL rem = std::floor(x);
    if(odd((int)rem))
        invert = !invert;
    rem = x - rem;
    if(rem > 0.5)
        rem = 1.0 - rem;
    if(rem == 0.5)
        rem = NumericTraits<REAL>::one();
    else
        rem = std::sin(M_PI * rem);
    return invert
              ? -rem
              : rem;
}

    /** \brief cos(pi*x).

        Essentially calls <tt>std::cos(M_PI*x)</tt> but uses a more accurate implementation
        to make sure that <tt>cos_pi(1.0) == -1.0</tt> and <tt>cos_pi(0.5) == 0.0</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class REAL,
          VIGRA_REQUIRE<std::is_floating_point<REAL>::value> >
REAL cos_pi(REAL x)
{
    return sin_pi(x+0.5);
}

/**********************************************************/
/*                                                        */
/*                     gamma(), loggamma()                */
/*                                                        */
/**********************************************************/

inline double gamma(double x)
{
    vigra_precondition(x <= 171.0,
        "gamma(): argument cannot exceed 171.0.");

    vigra_precondition(x > 0.0 || fmod(x, 1.0) != 0,
         "gamma(): gamma function is undefined for 0 and negative integers.");

    return tgamma(x);
}

inline double loggamma(double x)
{
    vigra_precondition(x > 0.0,
        "loggamma(): argument must be positive.");

    vigra_precondition(x <= 1.0e307,
        "loggamma(): argument must not exceed 1e307.");

    return lgamma(x);
}

/**********************************************************/
/*                                                        */
/*                        clipping                        */
/*                                                        */
/**********************************************************/

template <class T, class U=T>
inline
enable_if_t<std::is_arithmetic<T>::value && std::is_convertible<U, T>::value,
            T>
clipLower(T t, U lowerBound = U())
{
    return t < (T)lowerBound ? (T)lowerBound : t;
}

template <class T, class U=T>
inline
enable_if_t<std::is_arithmetic<T>::value && std::is_convertible<U, T>::value,
            T>
clipUpper(T t, U upperBound)
{
    return t > (T)upperBound ? (T)upperBound : t;
}

template <class T, class U=T>
inline
enable_if_t<std::is_arithmetic<T>::value && std::is_convertible<U, T>::value,
            T>
clip(T t, U lowerBound, U upperBound)
{
    return t < (T)lowerBound
              ? (T)lowerBound
              : t > (T)upperBound
                  ? (T)upperBound
                  : t;
}

/**********************************************************/
/*                                                        */
/*            scalar overloads of array functions         */
/*                                                        */
/**********************************************************/

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T
sum(T const t)
{
    return t;
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T
prod(T const t)
{
    return t;
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T
mean(T const t)
{
    return t;
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline bool
any(T const t)
{
    return t != T();
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline bool
all(T const t)
{
    return t != T();
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline bool
all_finite(T const t)
{
    return isfinite(t);
}

//@}

} // namespace vigra

#endif /* VIGRA2_MATHUTIL_HXX */
