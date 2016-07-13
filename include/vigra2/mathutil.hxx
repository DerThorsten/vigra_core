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

using std::acos;
using std::asin;
using std::atan;
using std::ceil;
using std::cos;
using std::exp;
using std::floor;
using std::log10;
using std::log;
using std::round;
using std::sin;
using std::sqrt;
using std::tan;
using std::atan2;
using std::fmod;
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
inline std::common_type_t<T1, T2>
min(T1 const & t1, T2 const & t2)
{
    return std::min<std::common_type_t<T1, T2>>(t1, t2);
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
inline std::common_type_t<T1, T2>
max(T1 const & t1, T2 const & t2)
{
    return std::max<typename std::common_type<T1, T2>::type>(t1, t2);
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
/*                     isinf(), isnan()                   */
/*                                                        */
/**********************************************************/

#ifndef _MSC_VER

using std::isinf;
using std::isnan;

#else

template <class REAL>
inline bool isinf(REAL v)
{
    return _finite(v) == 0;
}

template <class REAL>
inline bool isnan(REAL v)
{
    return _isnan(v) != 0;
}

#endif

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
/*                         round()                        */
/*                                                        */
/**********************************************************/

    // /** \brief The rounding function.

        // Defined for all floating point types. Rounds towards the nearest integer
        // such that <tt>abs(round(t)) == round(abs(t))</tt> for all <tt>t</tt>.

        // <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        // Namespace: vigra
    // */
// #ifdef DOXYGEN // only for documentation
// REAL round(REAL v);
// #endif

// inline float round(float t)
// {
     // return t >= 0.0
                // ? floor(t + 0.5f)
                // : ceil(t - 0.5f);
// }

// inline double round(double t)
// {
     // return t >= 0.0
                // ? floor(t + 0.5)
                // : ceil(t - 0.5);
// }

// inline long double round(long double t)
// {
     // return t >= 0.0
                // ? floor(t + 0.5)
                // : ceil(t - 0.5);
// }


    /** \brief Round and cast to integer.

        Rounds to the nearest integer like round(), but casts the result to
        <tt>int</tt> (this will be faster and is usually needed anyway).

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline int roundi(double t)
{
     return t >= 0.0
                ? int(t + 0.5)
                : int(t - 0.5);
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
/*                          hypot()                       */
/*                                                        */
/**********************************************************/

#ifdef VIGRA_NO_HYPOT
    /** \brief Compute the Euclidean distance (length of the hypotenuse of a right-angled triangle).

        The  hypot()  function  returns  the  sqrt(a*a  +  b*b).
        It is implemented in a way that minimizes round-off error.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double hypot(double a, double b)
{
    double absa = std::fabs(a), absb = std::fabs(b);
    if (absa > absb)
        return absa * std::sqrt(1.0 + sq(absb/absa));
    else
        return absb == 0.0
                   ? 0.0
                   : absb * std::sqrt(1.0 + sq(absa/absb));
}

#else

using std::hypot;

#endif

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

    /** \brief The binary sign function.

        Transfers the sign of \a t2 to \a t1.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2>
inline T1 sign(T1 t1, T2 t2)
{
    return t2 >= NumericTraits<T2>::zero()
               ? abs(t1)
               : -abs(t1);
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
/*                           erf()                        */
/*                                                        */
/**********************************************************/

#if defined(_MSC_VER) && _MSC_VER < 1800

namespace detail {

template <class T>
double erfImpl(T x)
{
    double t = 1.0/(1.0+0.5*std::fabs(x));
    double ans = t*std::exp(-x*x-1.26551223+t*(1.00002368+t*(0.37409196+
                                    t*(0.09678418+t*(-0.18628806+t*(0.27886807+
                                    t*(-1.13520398+t*(1.48851587+t*(-0.82215223+
                                    t*0.17087277)))))))));
    if (x >= 0.0)
        return 1.0 - ans;
    else
        return ans - 1.0;
}

} // namespace detail

    /** \brief The error function.

        If <tt>erf()</tt> is not provided in the C standard math library (as it should according to the
        new C99 standard ?), VIGRA implements <tt>erf()</tt> as an approximation of the error
        function

        \f[
            \mbox{erf}(x) = \int_0^x e^{-t^2} dt
        \f]

        according to the formula given in Press et al. "Numerical Recipes".

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double erf(double x)
{
    return detail::erfImpl(x);
}

#else

using std::erf;

#endif

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

namespace detail {

template <class REAL>
struct GammaImpl
{
    static REAL gamma(REAL x);
    static REAL loggamma(REAL x);

    static double g[];
    static double a[];
    static double t[];
    static double u[];
    static double v[];
    static double s[];
    static double r[];
    static double w[];
};

template <class REAL>
double GammaImpl<REAL>::g[] = {
    1.0,
    0.5772156649015329,
   -0.6558780715202538,
   -0.420026350340952e-1,
    0.1665386113822915,
   -0.421977345555443e-1,
   -0.9621971527877e-2,
    0.7218943246663e-2,
   -0.11651675918591e-2,
   -0.2152416741149e-3,
    0.1280502823882e-3,
   -0.201348547807e-4,
   -0.12504934821e-5,
    0.1133027232e-5,
   -0.2056338417e-6,
    0.6116095e-8,
    0.50020075e-8,
   -0.11812746e-8,
    0.1043427e-9,
    0.77823e-11,
   -0.36968e-11,
    0.51e-12,
   -0.206e-13,
   -0.54e-14,
    0.14e-14
};

template <class REAL>
double GammaImpl<REAL>::a[] = {
    7.72156649015328655494e-02,
    3.22467033424113591611e-01,
    6.73523010531292681824e-02,
    2.05808084325167332806e-02,
    7.38555086081402883957e-03,
    2.89051383673415629091e-03,
    1.19270763183362067845e-03,
    5.10069792153511336608e-04,
    2.20862790713908385557e-04,
    1.08011567247583939954e-04,
    2.52144565451257326939e-05,
    4.48640949618915160150e-05
};

template <class REAL>
double GammaImpl<REAL>::t[] = {
    4.83836122723810047042e-01,
    -1.47587722994593911752e-01,
    6.46249402391333854778e-02,
    -3.27885410759859649565e-02,
    1.79706750811820387126e-02,
    -1.03142241298341437450e-02,
    6.10053870246291332635e-03,
    -3.68452016781138256760e-03,
    2.25964780900612472250e-03,
    -1.40346469989232843813e-03,
    8.81081882437654011382e-04,
    -5.38595305356740546715e-04,
    3.15632070903625950361e-04,
    -3.12754168375120860518e-04,
    3.35529192635519073543e-04
};

template <class REAL>
double GammaImpl<REAL>::u[] = {
    -7.72156649015328655494e-02,
    6.32827064025093366517e-01,
    1.45492250137234768737e+00,
    9.77717527963372745603e-01,
    2.28963728064692451092e-01,
    1.33810918536787660377e-02
};

template <class REAL>
double GammaImpl<REAL>::v[] = {
    0.0,
    2.45597793713041134822e+00,
    2.12848976379893395361e+00,
    7.69285150456672783825e-01,
    1.04222645593369134254e-01,
    3.21709242282423911810e-03
};

template <class REAL>
double GammaImpl<REAL>::s[] = {
    -7.72156649015328655494e-02,
    2.14982415960608852501e-01,
    3.25778796408930981787e-01,
    1.46350472652464452805e-01,
    2.66422703033638609560e-02,
    1.84028451407337715652e-03,
    3.19475326584100867617e-05
};

template <class REAL>
double GammaImpl<REAL>::r[] = {
    0.0,
    1.39200533467621045958e+00,
    7.21935547567138069525e-01,
    1.71933865632803078993e-01,
    1.86459191715652901344e-02,
    7.77942496381893596434e-04,
    7.32668430744625636189e-06
};

template <class REAL>
double GammaImpl<REAL>::w[] = {
    4.18938533204672725052e-01,
    8.33333333333329678849e-02,
    -2.77777777728775536470e-03,
    7.93650558643019558500e-04,
    -5.95187557450339963135e-04,
    8.36339918996282139126e-04,
    -1.63092934096575273989e-03
};

template <class REAL>
REAL GammaImpl<REAL>::gamma(REAL x)
{
    int i, k, m, ix = (int)x;
    double ga = 0.0, gr = 0.0, r = 0.0, z = 0.0;

    vigra_precondition(x <= 171.0,
        "gamma(): argument cannot exceed 171.0.");

    if (x == ix)
    {
        if (ix > 0)
        {
            ga = 1.0;               // use factorial
            for (i=2; i<ix; ++i)
            {
               ga *= i;
            }
        }
        else
        {
            vigra_precondition(false,
                 "gamma(): gamma function is undefined for 0 and negative integers.");
        }
     }
     else
     {
        if (abs(x) > 1.0)
        {
            z = abs(x);
            m = (int)z;
            r = 1.0;
            for (k=1; k<=m; ++k)
            {
                r *= (z-k);
            }
            z -= m;
        }
        else
        {
            z = x;
        }
        gr = g[24];
        for (k=23; k>=0; --k)
        {
            gr = gr*z+g[k];
        }
        ga = 1.0/(gr*z);
        if (abs(x) > 1.0)
        {
            ga *= r;
            if (x < 0.0)
            {
                ga = -M_PI/(x*ga*sin_pi(x));
            }
        }
    }
    return ga;
}

/*
 * the following code is derived from lgamma_r() by Sun
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 */
template <class REAL>
REAL GammaImpl<REAL>::loggamma(REAL x)
{
    vigra_precondition(x > 0.0,
        "loggamma(): argument must be positive.");

    vigra_precondition(x <= 1.0e307,
        "loggamma(): argument must not exceed 1e307.");

    double res;

    if (x < 4.2351647362715017e-22)
    {
        res = -std::log(x);
    }
    else if ((x == 2.0) || (x == 1.0))
    {
        res = 0.0;
    }
    else if (x < 2.0)
    {
        const double tc  =  1.46163214496836224576e+00;
        const double tf  = -1.21486290535849611461e-01;
        const double tt  = -3.63867699703950536541e-18;
        if (x <= 0.9)
        {
            res = -std::log(x);
            if (x >= 0.7316)
            {
                double y = 1.0-x;
                double z = y*y;
                double p1 = a[0]+z*(a[2]+z*(a[4]+z*(a[6]+z*(a[8]+z*a[10]))));
                double p2 = z*(a[1]+z*(a[3]+z*(a[5]+z*(a[7]+z*(a[9]+z*a[11])))));
                double p  = y*p1+p2;
                res  += (p-0.5*y);
            }
            else if (x >= 0.23164)
            {
                double y = x-(tc-1.0);
                double z = y*y;
                double w = z*y;
                double p1 = t[0]+w*(t[3]+w*(t[6]+w*(t[9] +w*t[12])));
                double p2 = t[1]+w*(t[4]+w*(t[7]+w*(t[10]+w*t[13])));
                double p3 = t[2]+w*(t[5]+w*(t[8]+w*(t[11]+w*t[14])));
                double p  = z*p1-(tt-w*(p2+y*p3));
                res += (tf + p);
            }
            else
            {
                double y = x;
                double p1 = y*(u[0]+y*(u[1]+y*(u[2]+y*(u[3]+y*(u[4]+y*u[5])))));
                double p2 = 1.0+y*(v[1]+y*(v[2]+y*(v[3]+y*(v[4]+y*v[5]))));
                res += (-0.5*y + p1/p2);
            }
        }
        else
        {
            res = 0.0;
            if (x >= 1.7316)
            {
                double y = 2.0-x;
                double z = y*y;
                double p1 = a[0]+z*(a[2]+z*(a[4]+z*(a[6]+z*(a[8]+z*a[10]))));
                double p2 = z*(a[1]+z*(a[3]+z*(a[5]+z*(a[7]+z*(a[9]+z*a[11])))));
                double p  = y*p1+p2;
                res  += (p-0.5*y);
            }
            else if(x >= 1.23164)
            {
                double y = x-tc;
                double z = y*y;
                double w = z*y;
                double p1 = t[0]+w*(t[3]+w*(t[6]+w*(t[9] +w*t[12])));
                double p2 = t[1]+w*(t[4]+w*(t[7]+w*(t[10]+w*t[13])));
                double p3 = t[2]+w*(t[5]+w*(t[8]+w*(t[11]+w*t[14])));
                double p  = z*p1-(tt-w*(p2+y*p3));
                res += (tf + p);
            }
            else
            {
                double y = x-1.0;
                double p1 = y*(u[0]+y*(u[1]+y*(u[2]+y*(u[3]+y*(u[4]+y*u[5])))));
                double p2 = 1.0+y*(v[1]+y*(v[2]+y*(v[3]+y*(v[4]+y*v[5]))));
                res += (-0.5*y + p1/p2);
            }
        }
    }
    else if(x < 8.0)
    {
        double i = std::floor(x);
        double y = x-i;
        double p = y*(s[0]+y*(s[1]+y*(s[2]+y*(s[3]+y*(s[4]+y*(s[5]+y*s[6]))))));
        double q = 1.0+y*(r[1]+y*(r[2]+y*(r[3]+y*(r[4]+y*(r[5]+y*r[6])))));
        res = 0.5*y+p/q;
        double z = 1.0;
        while (i > 2.0)
        {
            --i;
            z *= (y+i);
        }
        res += std::log(z);
    }
    else if (x < 2.8823037615171174e+17)
    {
        double t = std::log(x);
        double z = 1.0/x;
        double y = z*z;
        double yy = w[0]+z*(w[1]+y*(w[2]+y*(w[3]+y*(w[4]+y*(w[5]+y*w[6])))));
        res = (x-0.5)*(t-1.0)+yy;
    }
    else
    {
        res =  x*(std::log(x) - 1.0);
    }

    return res;
}


} // namespace detail

    /** \brief The gamma function.

        This function implements the algorithm from<br>
        Zhang and Jin: "Computation of Special Functions", John Wiley and Sons, 1996.

        The argument must be <= 171.0 and cannot be zero or a negative integer. An
        exception is thrown when these conditions are violated.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double gamma(double x)
{
    return detail::GammaImpl<double>::gamma(x);
}

    /** \brief The natural logarithm of the gamma function.

        This function is based on a free implementation by Sun Microsystems, Inc., see
        <a href="http://www.sourceware.org/cgi-bin/cvsweb.cgi/~checkout~/src/newlib/libm/mathfp/er_lgamma.c?rev=1.6&content-type=text/plain&cvsroot=src">sourceware.org</a> archive. It can be removed once all compilers support the new C99
        math functions.

        The argument must be positive and < 1e30. An exception is thrown when these conditions are violated.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double loggamma(double x)
{
    return detail::GammaImpl<double>::loggamma(x);
}

/**********************************************************/
/*                                                        */
/*                        clipping                        */
/*                                                        */
/**********************************************************/

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T
clipLower(T const t, T const lowerBound = T())
{
    return t < lowerBound ? lowerBound : t;
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T
clipUpper(T const t, T const upperBound)
{
    return t > upperBound ? upperBound : t;
}

template <class T,
          VIGRA_REQUIRE<std::is_arithmetic<T>::value> >
inline T
clip(T const t, T const lowerBound, T const upperBound)
{
    return t < lowerBound
              ?lowerBound
              : t > upperBound
                  ? upperBound
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

//@}

} // namespace vigra

#endif /* VIGRA2_MATHUTIL_HXX */
