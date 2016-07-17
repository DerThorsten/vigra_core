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

#include <vigra2/config.hxx>
#include <vigra2/unittest.hxx>
#include <vigra2/mathutil.hxx>
#include <vigra2/array_nd.hxx>
#include <vigra2/array_math.hxx>
#include <numeric>  // std::iota()
#include <typeinfo>
#include <iostream>
#include <string>

namespace vigra {

template <class Ex>
enable_if_t<ArrayMathConcept<Ex>::value> 
testEx(Ex && ex)
{
    std::cerr << typeid(ex).name() << "\n";
}

template <int N>
struct ArrayMathTest
{
    typedef ArrayND<N, int>         Array;
    typedef ArrayND<N, double>      DArray;
    typedef ArrayViewND<N, int>     View;
    typedef TinyArray<int, 3>       Vector;
    typedef ArrayViewND<N, Vector>  VectorView;
    typedef ArrayND<N, Vector>      VArray;
    typedef Shape<N>                S;

    S s{ 4,3,2 };
    Array z, i, j;
    DArray a, b, c, d, r1, r2;

    ArrayMathTest()
        : z(s)
        , i(s)
        , j(s, 2)
        , a(s)
        , b(s)
        , c(s)
        , d(s)
        , r1(s)
        , r2(s)
    {
        std::iota(i.data(), i.data()+i.size(), 0);

        for (unsigned int k = 0; k < 24; ++k)
        {
            a[k] = std::exp(-0.2*k);
            b[k] = 0.5 + k;
            c[k] = 100.0 + k;
            d[k] = -(k + 1.0);
        }
    }

    void testResultType()
    {
        using namespace array_math;

        should((std::is_same<typename decltype(-i)::result_type, int>::value));
        should((std::is_same<typename decltype(-(-i))::result_type, int>::value));
        should((std::is_same<typename decltype(i + i)::result_type, int>::value));
        should((std::is_same<typename decltype(i + 1.0)::result_type, double>::value));
        should((std::is_same<typename decltype(1.0 + i)::result_type, double>::value));
        should((std::is_same<typename decltype(i + -i)::result_type, int>::value));
        should((std::is_same<typename decltype(-i + i)::result_type, int>::value));
        should((std::is_same<typename decltype(-i + -i)::result_type, int>::value));
        should((std::is_same<typename decltype(i + Shape<>())::result_type, Shape<>>::value));
    }

    void testUnary()
    {
        using namespace array_math;

        // basic checks of negation
        should(z == Array(-z));
        should(i == Array(-(-i)));

        Array ma = -i;
        Array fa(-i, F_ORDER);
        should(ma.shape() == fa.shape());
        should(ma.strides() != fa.strides());
        should(ma == fa);

        // check in-place write
        auto p = fa.data();
        fa = -fa;
        shouldEqual(p, fa.data());
        int count = 0;
        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k, ++count)
                {
                    shouldEqual((ma[{i, j, k}]), -count);
                    shouldEqual((fa[{i, j, k}]), count);
                }

        // check in-place write with incompatible strides (creates an internal copy of the RHS)
        static const int M = (N == runtime_size) ? runtime_size : 2;
        ArrayND<M, int> a2(Shape<2>{4, 4});
        std::iota(a2.data(), a2.data() + a2.size(), 0);

        ArrayND<M, int> ma2t(a2);
        p = ma2t.data();
        ma2t = -transpose(ma2t);
        shouldEqual(p, ma2t.data());
        count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
            {
                shouldEqual((a2[{i, j}]), count);
                shouldEqual((ma2t[{j, i}]), -count);
            }

        transpose(ma2t) = -ma2t;
        shouldEqual(p, ma2t.data());
        should(a2 == ma2t);

        // check that 'bool' is replaced by 'unsigned char'
        should((std::is_same<typename decltype(!i)::result_type, unsigned char>::value));

        // check all unary functions

#define VIGRA_TEST_UNARY_FUNCTION(FCT, RHS) \
        r1 = FCT(RHS); \
        for(int k=0; k<r2.size(); ++k) \
            r2[k] = FCT(RHS[k]); \
        should(r2 == r1)

        VIGRA_TEST_UNARY_FUNCTION(-, b);
        VIGRA_TEST_UNARY_FUNCTION(!, i);
        VIGRA_TEST_UNARY_FUNCTION(~, i);

        VIGRA_TEST_UNARY_FUNCTION(abs, d);
        VIGRA_TEST_UNARY_FUNCTION(abs, d);

        VIGRA_TEST_UNARY_FUNCTION(cos, b);
        VIGRA_TEST_UNARY_FUNCTION(sin, b);
        VIGRA_TEST_UNARY_FUNCTION(tan, b);
        VIGRA_TEST_UNARY_FUNCTION(sin_pi, b);
        VIGRA_TEST_UNARY_FUNCTION(cos_pi, b);
        VIGRA_TEST_UNARY_FUNCTION(acos, a);
        VIGRA_TEST_UNARY_FUNCTION(asin, a);
        VIGRA_TEST_UNARY_FUNCTION(atan, b);

        VIGRA_TEST_UNARY_FUNCTION(cosh, b);
        VIGRA_TEST_UNARY_FUNCTION(sinh, b);
        VIGRA_TEST_UNARY_FUNCTION(tanh, b);
        VIGRA_TEST_UNARY_FUNCTION(acosh, c);
        VIGRA_TEST_UNARY_FUNCTION(asinh, a);
        VIGRA_TEST_UNARY_FUNCTION(atanh, a);

        VIGRA_TEST_UNARY_FUNCTION(sqrt, b);
        VIGRA_TEST_UNARY_FUNCTION(cbrt, b);
        VIGRA_TEST_UNARY_FUNCTION(sqrti, i);
        VIGRA_TEST_UNARY_FUNCTION(sq, b);

        VIGRA_TEST_UNARY_FUNCTION(exp, b);
        VIGRA_TEST_UNARY_FUNCTION(exp2, b);
        VIGRA_TEST_UNARY_FUNCTION(expm1, b);
        VIGRA_TEST_UNARY_FUNCTION(log, b);
        VIGRA_TEST_UNARY_FUNCTION(log2, b);
        VIGRA_TEST_UNARY_FUNCTION(log10, b);
        VIGRA_TEST_UNARY_FUNCTION(log1p, b);
        VIGRA_TEST_UNARY_FUNCTION(logb, b);
        VIGRA_TEST_UNARY_FUNCTION(ilogb, b);

        VIGRA_TEST_UNARY_FUNCTION(ceil, b);
        VIGRA_TEST_UNARY_FUNCTION(floor, b);
        VIGRA_TEST_UNARY_FUNCTION(round, b);
        VIGRA_TEST_UNARY_FUNCTION(lround, b);
        VIGRA_TEST_UNARY_FUNCTION(llround, b);
        VIGRA_TEST_UNARY_FUNCTION(roundi, b);
        VIGRA_TEST_UNARY_FUNCTION(even, i);
        VIGRA_TEST_UNARY_FUNCTION(odd, i);
        VIGRA_TEST_UNARY_FUNCTION(sign, d);
        VIGRA_TEST_UNARY_FUNCTION(signi, b);

        VIGRA_TEST_UNARY_FUNCTION(erf, b);
        VIGRA_TEST_UNARY_FUNCTION(erfc, b);
        VIGRA_TEST_UNARY_FUNCTION(tgamma, b);
        VIGRA_TEST_UNARY_FUNCTION(lgamma, b);

        r1 = elementwiseNorm(d);
        r2 = elementwiseSquaredNorm(d);
        for (int k = 0; k < r2.size(); ++k)
        {
            shouldEqual(r1[k], norm(d[k]));
            shouldEqual(r2[k], squaredNorm(d[k]));
        }

        should(any(i));
        shouldNot(any(z));
        shouldNot(any(i - i));
        z + 1;
        any(z + 1);
        should(any(z+1));

        should(all_finite(z));
        shouldNot(all_finite(1.0 / z));

        i += 1;
        should(all(i));
        shouldNot(all(z));
        shouldNot(all(i - 1));
        should(all(z+1));

        shouldEqual(sum(-i), -300);
        shouldEqualTolerance(prod(-i, 1.0), 6.204484017332394e+23, 1e-13);
    }

    void testBinary()
    {
        using namespace array_math;

        Array mpa = -i + i;
        should(z == mpa);
        Array pma = -1 + z + 1;
        should(z == pma);

#define VIGRA_TEST_BINARY_FUNCTION(FCT, V1, V2) \
        r1 = FCT(V1, V2); \
        for(int k=0; k<r2.size(); ++k) \
            r2[k] = FCT(V1[k], V2[k]); \
        should(r2 == r1)

#define VIGRA_TEST_BINARY_OPERATOR(OP, V1, V2) \
        r1 = V1 OP V2; \
        for(int k=0; k<r2.size(); ++k) \
            r2[k] = V1[k] OP V2[k]; \
        should(r2 == r1)

        // FIXME: explicit namespace qualification vigra:: should not be necessary
        //        (somehow, MSVC otherwise calls std::min and std::max, although these
        //         templates are not supposed to be visible in the present namespace
        //         without explicit qualification std::)
        VIGRA_TEST_BINARY_FUNCTION(vigra::min, b, c);
        VIGRA_TEST_BINARY_FUNCTION(vigra::max, b, c);
        VIGRA_TEST_BINARY_FUNCTION(fmin, b, c);
        VIGRA_TEST_BINARY_FUNCTION(fmax, b, c);

        VIGRA_TEST_BINARY_FUNCTION(atan2, b, c);
        VIGRA_TEST_BINARY_FUNCTION(clipLower, b, z);
        VIGRA_TEST_BINARY_FUNCTION(clipUpper, b, z);
        VIGRA_TEST_BINARY_FUNCTION(copysign, b, d);
        VIGRA_TEST_BINARY_FUNCTION(fdim, b, d);
        VIGRA_TEST_BINARY_FUNCTION(fmod, b, c);
        VIGRA_TEST_BINARY_FUNCTION(hypot, b, c);
        VIGRA_TEST_BINARY_FUNCTION(pow, b, c);

        VIGRA_TEST_BINARY_OPERATOR(+, b, c);
        VIGRA_TEST_BINARY_OPERATOR(-, b, c);
        VIGRA_TEST_BINARY_OPERATOR(*, b, c);
        VIGRA_TEST_BINARY_OPERATOR(/ , b, c);
        VIGRA_TEST_BINARY_OPERATOR(%, i, j);
        VIGRA_TEST_BINARY_OPERATOR(&&, i, j);
        VIGRA_TEST_BINARY_OPERATOR(|| , i, j);
        VIGRA_TEST_BINARY_OPERATOR(<, i, j);
        VIGRA_TEST_BINARY_OPERATOR(<= , i, j);
        VIGRA_TEST_BINARY_OPERATOR(>, i, j);
        VIGRA_TEST_BINARY_OPERATOR(>= , i, j);
        VIGRA_TEST_BINARY_OPERATOR(<< , i, j);
        VIGRA_TEST_BINARY_OPERATOR(>> , i, j);
        VIGRA_TEST_BINARY_OPERATOR(&, i, j);
        VIGRA_TEST_BINARY_OPERATOR(| , i, j);
        VIGRA_TEST_BINARY_OPERATOR(^, i, j);

        r1 = elementwiseEqual(i, j);
        r2 = elementwiseNotEqual(i, j);
        for (int k = 0; k < r2.size(); ++k)
        {
            shouldEqual(r1[k], double(i[k] == j[k]));
            shouldEqual(r2[k], double(i[k] != j[k]));
        }

        r1 = sqrt(b)*(d + 2.0);
        std::transform(b.data(), b.data() + b.size(), d.data(), r2.data(),
            [](double u, double v) {
            return sqrt(u)*(v + 2.0);
        });
        should(r1 == r2);

        r1 = (b-d)*2.0*(abs(d)/b);
        std::transform(b.data(), b.data() + b.size(), d.data(), r2.data(),
            [](double u, double v) {
            return (u-v)*2.0*(abs(v)/u);
        });
        should(r1 == r2);
    }

    void testArithmeticAssignment()
    {
        using namespace array_math;

        testEx(2 * i);

        i += 1;
        Array t(i);

        t -= 2 * i;
        should(-i == t);
        should(t == -i);

        t /= -i;
        should(t == 1);
        should(1 == t);

        t *= -i;
        should(t + 1 == -i + 1);

        t += -t;
        should(!any(t));
        should(!any(t * t));
        should(!any(t * -t));
        should(!any(t * 2));
        should(!any(2 * t));
    }

    void testOverlappingMemory()
    {
        using namespace array_math;

        const int M = (N == runtime_size)
            ? N
            : 2;
        ArrayViewND<M, int> vs(Shape<M>{4, 4}, i.data());
        ArrayND<M, int> v1 = vs;
        v1 = -v1;

        int count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                shouldEqual((v1[{i, j}]), -count);

        v1 = -v1.transpose();

        count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                shouldEqual((v1[{j, i}]), count);

        ArrayND<M, int> v2(vs, F_ORDER);
        v2 = -v2;

        count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                shouldEqual((v2[{i, j}]), -count);

        v2 = -v2.transpose();

        count = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j, ++count)
                shouldEqual((v2[{j, i}]), count);

        {
            v1 = vs;
            auto b1 = v1.bind(0, 1),
                 b2 = v1.bind(0, 2);
            b1 = -b2;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (i == 1)
                        shouldEqual((-vs[{2, j}]), (v1[{i, j}]));
                    else
                        shouldEqual((vs[{i, j}]), (v1[{i, j}]));
        }
        {
            v1 = vs;
            auto b1 = v1.bind(1, 1),
                 b2 = v1.bind(1, 2);
            b1 = -b2;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (j == 1)
                        shouldEqual((-vs[{i, 2}]), (v1[{i, j}]));
                    else
                        shouldEqual((vs[{i, j}]), (v1[{i, j}]));
        }
        {
            v1 = vs;
            auto b1 = v1.bind(1, 1),
                 b2 = v1.bind(1, 2);
            b2 = -b1;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (j == 2)
                        shouldEqual((-vs[{i, 1}]), (v1[{i, j}]));
                    else
                        shouldEqual((vs[{i, j}]), (v1[{i, j}]));
        }
        {
            v1 = vs;
            auto b1 = v1.bind(0, 2),
                 b2 = v1.bind(1, 1);
            b1 = -b2;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (i == 2)
                        shouldEqual((-vs[{j, 1}]), (v1[{i, j}]));
                    else
                        shouldEqual((vs[{i, j}]), (v1[{i, j}]));
        }
        {
            v1 = vs;
            auto b1 = v1.bind(0, 1),
                 b2 = v1.bind(1, 2);
            b2 = -b1;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (j == 2)
                        shouldEqual((-vs[{1, i}]), (v1[{i, j}]));
                    else
                        shouldEqual((vs[{i, j}]), (v1[{i, j}]));
        }
    }

    void testVectorTypes()
    {
        using namespace array_math;

        VArray va = mgrid(s);

        shouldEqual(va.shape(), s);

        for (int i = 0; i < s[0]; ++i)
            for (int j = 0; j < s[1]; ++j)
                for (int k = 0; k < s[2]; ++k)
                {
                    shouldEqual((va[{i, j, k}]), (S{ i,j,k }));
                }

        // FIXME: add tests for arrays with complex numbers
    }
};

struct ArrayMathTestSuite
    : public test_suite
{
    ArrayMathTestSuite()
        : test_suite("ArrayMathTestSuite")
    {
        addTests<3>();
        addTests<runtime_size>();
    }

    template <int N>
    void addTests()
    {
        add(testCase(&ArrayMathTest<N>::testResultType));
        add(testCase(&ArrayMathTest<N>::testUnary));
        add(testCase(&ArrayMathTest<N>::testBinary));
        add(testCase(&ArrayMathTest<N>::testArithmeticAssignment));
        add(testCase(&ArrayMathTest<N>::testOverlappingMemory));
        add(testCase(&ArrayMathTest<N>::testVectorTypes)); 
    }
};

} // namespace vigra

int main(int argc, char ** argv)
{
    vigra::ArrayMathTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
