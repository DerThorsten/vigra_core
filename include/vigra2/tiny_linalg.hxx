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

#ifndef VIGRA2_TINY_LINALG_HXX
#define VIGRA2_TINY_LINALG_HXX

#include "config.hxx"
#include "numeric_traits.hxx"
#include "tinyarray.hxx"
#include "mathutil.hxx"

namespace vigra {

/** \addtogroup MathFunctions
*/
//@{

    // vector-matrix product
template <class V1, class D1, class V2, class D2, int N1, int N2>
inline
TinyArray<PromoteType<V1, V2>, N2>
dot(TinyArrayBase<V1, D1, N1> const & l,
    TinyArrayBase<V2, D2, N1, N2> const & r)
{
    TinyArray<PromoteType<V1, V2>, N2> res;
    for(int j=0; j < N2; ++j)
    {
        res[j] = l[0] * r(0,j);
        for(int i=1; i < N1; ++i)
            res[j] += l[i] * r(i,j);
    }
    return res;
}

    // vector - symmetric matrix product
template <class V1, class D1, class V2, int N>
inline
TinyArray<PromoteType<V1, V2>, N>
dot(TinyArrayBase<V1, D1, N> const & l,
    TinySymmetricView<V2, N> const & r)
{
    TinyArray<PromoteType<V1, V2>, N> res;
    for(int j=0; j < N; ++j)
    {
        res[j] = l[0] * r(0,j);
        for(int i=1; i < N; ++i)
            res[j] += l[i] * r(i,j);
    }
    return res;
}

    // matrix-vector product
template <class V1, class D1, class V2, class D2, int N1, int N2>
inline
TinyArray<PromoteType<V1, V2>, N1>
dot(TinyArrayBase<V1, D1, N1, N2> const & l,
    TinyArrayBase<V2, D2, N2> const & r)
{
    TinyArray<PromoteType<V1, V2>, N1> res;
    for(int i=0; i < N1; ++i)
    {
        res[i] = l(i,0) * r[0];
        for(int j=1; j < N2; ++j)
            res[i] += l(i,j) * r[j];
    }
    return res;
}

    // symmetric matrix - vector product
template <class V1, class V2, class D2, int N>
inline
TinyArray<PromoteType<V1, V2>, N>
dot(TinySymmetricView<V1, N> const & l,
    TinyArrayBase<V2, D2, N> const & r)
{
    TinyArray<PromoteType<V1, V2>, N> res;
    for(int i=0; i < N; ++i)
    {
        res[i] = l(i,0) * r[0];
        for(int j=1; j < N; ++j)
            res[i] += l(i,j) * r[j];
    }
    return res;
}

    // matrix-matrix product
template <class V1, class D1, class V2, class D2,
          int N1, int N2, int N3>
inline
TinyArray<PromoteType<V1, V2>, N1, N3>
dot(TinyArrayBase<V1, D1, N1, N2> const & l,
    TinyArrayBase<V2, D2, N2, N3> const & r)
{
    TinyArray<PromoteType<V1, V2>, N1, N3> res;
    for(int i=0; i < N1; ++i)
    {
        for(int j=0; j < N3; ++j)
        {
            res(i,j) = l(i,0) * r(0,j);
            for(int k=1; k < N2; ++k)
                res(i,j) += l(i,k) * r(k,j);
        }
    }
    return res;
}

    // matrix - symmetric matrix product
template <class V1, class D1, class V2,
          int N1, int N2>
inline
TinyArray<PromoteType<V1, V2>, N1, N2>
dot(TinyArrayBase<V1, D1, N1, N2> const & l,
    TinySymmetricView<V2, N2> const & r)
{
    TinyArray<PromoteType<V1, V2>, N1, N2> res;
    for(int i=0; i < N1; ++i)
    {
        for(int j=0; j < N2; ++j)
        {
            res(i,j) = l(i,0) * r(0,j);
            for(int k=1; k < N2; ++k)
                res(i,j) += l(i,k) * r(k,j);
        }
    }
    return res;
}

    // matrix - symmetric matrix product
template <class V1, class D1, class V2,
          int N1, int N2>
inline
TinyArray<PromoteType<V1, V2>, N1, N2>
dot(TinySymmetricView<V2, N1> const & l,
    TinyArrayBase<V1, D1, N1, N2> const & r)
{
    TinyArray<PromoteType<V1, V2>, N1, N2> res;
    for(int i=0; i < N1; ++i)
    {
        for(int j=0; j < N2; ++j)
        {
            res(i,j) = l(i,0) * r(0,j);
            for(int k=1; k < N1; ++k)
                res(i,j) += l(i,k) * r(k,j);
        }
    }
    return res;
}

    // symmetric matrix - symmetric matrix product
template <class V1, class V2, int N>
inline
TinyArray<PromoteType<V1, V2>, N*(N+1)/2>
dot(TinySymmetricView<V1, N> const & l,
    TinySymmetricView<V2, N> const & r)
{
    TinyArray<PromoteType<V1, V2>, N*(N+1)/2> res;
    TinySymmetricView<PromoteType<V1, V2>, N> view(res.data());
    for(int i=0; i < N; ++i)
    {
        for(int j=i; j < N; ++j)
        {
            view(i,j) = l(i,0) * r(0,j);
            for(int k=1; k < N; ++k)
                view(i,j) += l(i,k) * r(k,j);
        }
    }
    return res;
}

    /** \brief Compute the eigenvalues of a 2x2 real symmetric matrix.

        This uses the analytical eigenvalue formula
        \f[
           \lambda_{1,2} = \frac{1}{2}\left(a_{00} + a_{11} \pm \sqrt{(a_{00} - a_{11})^2 + 4 a_{01}^2}\right)
        \f]

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T>
void symmetric2x2Eigenvalues(T a00, T a01, T a11, T * r0, T * r1)
{
    double d  = hypot(a00 - a11, 2.0*a01);
    *r0 = static_cast<T>(0.5*(a00 + a11 + d));
    *r1 = static_cast<T>(0.5*(a00 + a11 - d));
    if(*r0 < *r1)
        std::swap(*r0, *r1);
}

    /** \brief Compute the eigenvalues of a 3x3 real symmetric matrix.

        This uses a numerically stable version of the analytical eigenvalue formula according to
        <p>
        David Eberly: <a href="http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf">
        <em>"Eigensystems for 3 × 3 Symmetric Matrices (Revisited)"</em></a>, Geometric Tools Documentation, 2006

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T>
void symmetric3x3Eigenvalues(T a00, T a01, T a02, T a11, T a12, T a22,
                             T * r0, T * r1, T * r2)
{
    double inv3 = 1.0 / 3.0, root3 = std::sqrt(3.0);

    double c0 = a00*a11*a22 + 2.0*a01*a02*a12 - a00*a12*a12 - a11*a02*a02 - a22*a01*a01;
    double c1 = a00*a11 - a01*a01 + a00*a22 - a02*a02 + a11*a22 - a12*a12;
    double c2 = a00 + a11 + a22;
    double c2Div3 = c2*inv3;
    double aDiv3 = (c1 - c2*c2Div3)*inv3;
    if (aDiv3 > 0.0)
        aDiv3 = 0.0;
    double mbDiv2 = 0.5*(c0 + c2Div3*(2.0*c2Div3*c2Div3 - c1));
    double q = mbDiv2*mbDiv2 + aDiv3*aDiv3*aDiv3;
    if (q > 0.0)
        q = 0.0;
    double magnitude = std::sqrt(-aDiv3);
    double angle = std::atan2(std::sqrt(-q),mbDiv2)*inv3;
    double cs = std::cos(angle);
    double sn = std::sin(angle);
    *r0 = static_cast<T>(c2Div3 + 2.0*magnitude*cs);
    *r1 = static_cast<T>(c2Div3 - magnitude*(cs + root3*sn));
    *r2 = static_cast<T>(c2Div3 - magnitude*(cs - root3*sn));
    if(*r0 < *r1)
        std::swap(*r0, *r1);
    if(*r0 < *r2)
        std::swap(*r0, *r2);
    if(*r1 < *r2)
        std::swap(*r1, *r2);
}

    /** \brief Compute the eigenvalues of a 2x2 real symmetric matrix.

        The matrix is specified by a <tt>TinySymmetricView</tt>,
        which only stores the upper triangular part. The algorithm uses the
        analytical eigenvalue formulas .

        <b>\#include</b> \<vigra2/tiny_linalg.hxx\><br>
        Namespace: vigra
    */
template <class T>
inline TinyArray<T, 2>
symmetricEigenvalues(TinySymmetricView<T, 2> const & a)
{
    TinyArray<T, 2> res(DontInit);
    symmetric2x2Eigenvalues(a[0], a[1], a[2], &res[0], &res[1]);
    return res;
}

    /** \brief Compute the eigenvalues of a 3x3 real symmetric matrix.

        The matrix is specified by a <tt>TinySymmetricView</tt>,
        which only stores the upper triangular part.
        The algorithm uses a numerically stable version of the
        analytical eigenvalue formula according to
        <p>
        David Eberly: <a href="http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf">
        <em>"Eigensystems for 3 × 3 Symmetric Matrices (Revisited)"</em></a>, Geometric Tools Documentation, 2006

        <b>\#include</b> \<vigra2/tiny_linalg.hxx\><br>
        Namespace: vigra
    */
template <class T>
inline TinyArray<T, 3>
symmetricEigenvalues(TinySymmetricView<T, 3> const & a)
{
    TinyArray<T, 3> res(DontInit);
    symmetric3x3Eigenvalues(a[0], a[1], a[2], a[3], a[4], a[5],
                            &res[0], &res[1], &res[2]);
    return res;
}

//@}

// FIXME: compute TinySymmetricView eigenvectors

} // namespace vigra

#endif // VIGRA2_TINY_LINALG_HXX
