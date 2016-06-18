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

#ifndef VIGRA2_CONCEPTS_HXX
#define VIGRA2_CONCEPTS_HXX

#include "config.hxx"
#include <type_traits>

namespace vigra {

enum SkipInitialization { DontInit };
enum ReverseCopyTag { ReverseCopy };
enum MemoryOrder { C_ORDER, F_ORDER };

template <bool CONCEPTS, class RETURN=void>
using EnableIf = typename std::enable_if<CONCEPTS, RETURN>::type;

/**********************************************************/
/*                                                        */
/*            multi-dimensional array concept             */
/*                                                        */
/**********************************************************/

struct ArrayNDTag {};

    // By default, 'ARRAY' fulfills the ArrayNDConcept if it contains
    // an embedded type 'typedef ArrayNDTag array_concept'.
    //
    // Alternatively, one can partially specialize ArrayNDConcept.
template <class ARRAY>
struct ArrayNDConcept
{
    static void * test(...);

    template <class U>
    static typename U::array_concept * test(U*, typename U::array_concept * = 0);

    static const bool value =
        std::is_same<decltype(test((ARRAY*)0)), ArrayNDTag*>::value;
};

template <class ARRAY, class RETURN=void>
using EnableIfArrayND =
      typename std::enable_if<ArrayNDConcept<ARRAY>::value, RETURN>::type;

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
struct IsIterator
{
    static char test(...);

    template <class U>
    static int test(U*, typename U::iterator_category * = 0);

    static const bool value =
        std::is_array<T>::value ||
        std::is_pointer<T>::value ||
        std::is_same<decltype(test((T*)0)), int>::value;
};

template <class T, class RETURN=void>
using EnableIfIterator =
      typename std::enable_if<IsIterator<T>::value, RETURN>::type;

} // namespace vigra

#endif // VIGRA2_CONCEPTS_HXX
