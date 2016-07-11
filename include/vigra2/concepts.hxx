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

#ifndef VIGRA2_CONCEPTS_HXX
#define VIGRA2_CONCEPTS_HXX

#include "config.hxx"
#include <type_traits>

namespace vigra {

    /// use biggest signed type for array indexing
using ArrayIndex = std::ptrdiff_t;

    /// constants to specialize templates whose size/ndim is only known at runtime
static const int runtime_size  = -1;
static const int runtime_ndim  = -1;
static const int runtime_order = -1;

enum SkipInitialization { DontInit };
enum ReverseCopyTag { ReverseCopy };
enum MemoryOrder { C_ORDER = 1, F_ORDER = 2 };

/**********************************************************/
/*                                                        */
/*                     concept checking                   */
/*                                                        */
/**********************************************************/

using std::enable_if;
using std::enable_if_t;

struct require_ok {};

template <bool CONCEPTS>
using ConceptCheck = typename enable_if<CONCEPTS, require_ok>::type;

#define VIGRA_REQUIRE typename Require = ConceptCheck

/**********************************************************/
/*                                                        */
/*            multi-dimensional array concept             */
/*                                                        */
/**********************************************************/

struct TinyArrayTag {};

    // TinyArrayConcept refers to TinyArrayBase and TinyArray.
    // By default, 'ARRAY' fulfills the TinyArrayConcept if it is derived
    // from TinyArrayTag.
    //
    // Alternatively, one can partially specialize TinyArrayConcept.
template <class ARRAY>
struct TinyArrayConcept
{
    static const bool value =std::is_base_of<TinyArrayTag, ARRAY>::value;
};

/**********************************************************/
/*                                                        */
/*            multi-dimensional array concept             */
/*                                                        */
/**********************************************************/

struct PointerNDTag {};

    // PointerNDConcept refers to the low-level multi-dimensional array API.
    // By default, 'ARRAY' fulfills the PointerNDConcept if it is derived
    // from PointerNDTag.
    //
    // Alternatively, one can partially specialize PointerNDConcept.
template <class ARRAY>
struct PointerNDConcept
{
    static const bool value = std::is_base_of<PointerNDTag, ARRAY>::value;
};

struct ArrayNDTag {};

    // ArrayNDConcept refers to the high-level multi-dimensional array API.
    // By default, 'ARRAY' fulfills the ArrayNDConcept if it is derived
    // from 'ArrayNDTag'.
    //
    // Alternatively, one can partially specialize ArrayNDConcept.
template <class ARRAY>
struct ArrayNDConcept
{
    static const bool value = std::is_base_of<ArrayNDTag, ARRAY>::value;
};

struct ArrayMathTag
: public PointerNDTag
{};

template <class ARRAY>
struct ArrayMathConcept
{
    static const bool value = std::is_base_of<ArrayMathTag, ARRAY>::value;
};

template <int N, int M>
struct CompatibleDimensions
{
    static const bool value = N == M || N == runtime_size || M == runtime_size;
};

/**********************************************************/
/*                                                        */
/*            check if a class is an iterator             */
/*                                                        */
/**********************************************************/

    // currently, we apply ony the simple rule that class T
    // must be a pointer or array or has an embedded typedef
    // 'iterator_category'. More sophisticated checks should
    // be added when needed.
template <class T>
struct IteratorConcept
{
    static char test(...);

    template <class U>
    static int test(U*, typename U::iterator_category * = 0);

    static const bool value =
        std::is_array<T>::value ||
        std::is_pointer<T>::value ||
        std::is_same<decltype(test((T*)0)), int>::value;
};

} // namespace vigra

#endif // VIGRA2_CONCEPTS_HXX
