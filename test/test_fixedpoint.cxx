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

#include <typeinfo>
#include <iostream>
#include <string>
#include <vigra2/unittest.hxx>
#include <vigra2/fixedpoint.hxx>

using namespace vigra;

struct FixedPointTest
{
    void testConstruction()
    {
        shouldEqual(fixedPoint(3).value, 3);
        shouldEqual(fixedPoint(-3).value, -3);
        shouldEqual(-fixedPoint(3).value, -3);

        shouldEqual((FixedPoint<3,4>(3).value), 3 << 4);
        shouldEqual((FixedPoint<3,4>(-3).value), -3 << 4);
        shouldEqual((-FixedPoint<3,4>(3).value), -3 << 4);

        shouldEqual((FixedPoint<3,4>(3.5).value), 56);
        shouldEqual((FixedPoint<3,4>(-3.5).value), -56);
        shouldEqual((-FixedPoint<3,4>(3.5).value), -56);

        try { FixedPoint<1, 8>(3.75); failTest("No exception thrown"); } catch(ContractViolation &) {}

        shouldEqual((NumericTraits<FixedPoint<1, 8> >::zero()).value, 0);
        shouldEqual((NumericTraits<FixedPoint<1, 8> >::one()).value, 1 << 8);
        shouldEqual((NumericTraits<FixedPoint<1, 8> >::max()).value, (1 << 9) - 1);
        shouldEqual((NumericTraits<FixedPoint<1, 8> >::min()).value, -((1 << 9) - 1));

        FixedPoint<2, 8> v(3.75);
        shouldEqual((FixedPoint<2, 8>(v).value), 15 << 6);
        shouldEqual((FixedPoint<3, 10>(v).value), 15 << 8);
        shouldEqual((FixedPoint<2, 2>(v).value), 15);
        shouldEqual((FixedPoint<2, 0>(v).value), 4);

        shouldEqual((FixedPoint<2, 8>(-v).value), -15 << 6);
        shouldEqual((FixedPoint<3, 10>(-v).value), -15 << 8);
        shouldEqual((FixedPoint<2, 2>(-v).value), -15);
        shouldEqual((FixedPoint<2, 0>(-v).value), -4);

        shouldEqual(fixed_point_cast<double>(v), 3.75);
        should((frac(v) == FixedPoint<0, 8>(0.75)));
        should((dual_frac(v) == FixedPoint<0, 8>(0.25)));
        should(floor(v) == 3);
        should(ceil(v) == 4);
        should(round(v) == 4);
        should(abs(v) == v);
        should((frac(-v) == FixedPoint<0, 8>(0.25)));
        should((dual_frac(-v) == FixedPoint<0, 8>(0.75)));
        should(floor(-v) == -4);
        should(ceil(-v) == -3);
        should(round(-v) == -4);
        should(abs(-v) == v);
        should(norm(-v) == v);
        should(squaredNorm(-v) == v*v);

        FixedPoint<3, 10> v1;
        shouldEqual((v1 = v).value, 15 << 8);
        shouldEqual((v1 = -v).value, -15 << 8);

        FixedPoint<2, 0> v2;
        shouldEqual((v2 = v).value, 4);
        shouldEqual((v2 = -v).value, -4);
    }

    void testComparison()
    {
        FixedPoint<3, 8> v1(3.75), v2(4);
        FixedPoint<2, 2> v3(3.75);
        should(v1 == v1);
        should(v1 == v3);
        should(!(v1 != v1));
        should(!(v1 != v3));
        should(v1 <= v1);
        should(v1 <= v3);
        should(!(v1 < v1));
        should(!(v1 < v3));
        should(v1 >= v1);
        should(v1 >= v3);
        should(!(v1 > v1));
        should(!(v1 > v3));

        should(v2 != v1);
        should(v2 != v3);
        should(!(v2 == v1));
        should(!(v2 == v3));
        should(!(v2 <= v1));
        should(!(v2 <= v3));
        should(!(v2 < v1));
        should(!(v2 < v3));
        should(v2 >= v1);
        should(v2 >= v3);
        should(v2 > v1);
        should(v2 > v3);
    }

    void testArithmetic()
    {
        FixedPoint<1, 16> t1(0.75), t2(0.25);
        signed char v1 = 1, v2 = 2, v4 = 4, v8 = 8;

        should((FixedPoint<1, 16>(t1) += t1) == (FixedPoint<1, 16>(1.5)));
        should((FixedPoint<1, 16>(t1) -= t1) == (FixedPoint<1, 16>(0.0)));
        should((FixedPoint<2, 16>(t1) *= t1) == (FixedPoint<1, 16>(9.0 / 16.0)));

        should(--t1 == (FixedPoint<1, 16>(-0.25)));
        should(t1 == (FixedPoint<1, 16>(-0.25)));
        should(++t1 == (FixedPoint<1, 16>(0.75)));
        should(t1 == (FixedPoint<1, 16>(0.75)));
        should(t1++ == (FixedPoint<1, 16>(0.75)));
        should(t1 == (FixedPoint<1, 16>(1.75)));
        should(t1-- == (FixedPoint<1, 16>(1.75)));
        should(t1 == (FixedPoint<1, 16>(0.75)));

        shouldEqual((t1 * fixedPoint(v1)).value, 3 << 14);
        shouldEqual((t2 * fixedPoint(v1)).value, 1 << 14);
        shouldEqual((-t1 * fixedPoint(v1)).value, -3 << 14);
        shouldEqual((-t2 * fixedPoint(v1)).value, -1 << 14);
        shouldEqual((t1 * -fixedPoint(v1)).value, -3 << 14);
        shouldEqual((t2 * -fixedPoint(v1)).value, -1 << 14);

        shouldEqual((FixedPoint<8, 2>(t1 * fixedPoint(v1))).value, 3);
        shouldEqual((FixedPoint<8, 2>(t2 * fixedPoint(v1))).value, 1);
        shouldEqual((FixedPoint<8, 2>(-t1 * fixedPoint(v1))).value, -3);
        shouldEqual((FixedPoint<8, 2>(-t2 * fixedPoint(v1))).value, -1);

        shouldEqual(floor(t1 * fixedPoint(v1) + t2 * fixedPoint(v2)), 1);
        shouldEqual(ceil(t1 * fixedPoint(v1) + t2 * fixedPoint(v2)), 2);
        shouldEqual(round(t1 * fixedPoint(v1) + t2 * fixedPoint(v2)), 1);
        shouldEqual(floor(t1 * fixedPoint(v4) + t2 * fixedPoint(v8)), 5);
        shouldEqual(ceil(t1 * fixedPoint(v4) + t2 * fixedPoint(v8)), 5);
        shouldEqual(round(t1 * fixedPoint(v4) + t2 * fixedPoint(v8)), 5);

        shouldEqual(floor(t1 * -fixedPoint(v1) - t2 * fixedPoint(v2)), -2);
        shouldEqual(ceil(t1 * -fixedPoint(v1) - t2 * fixedPoint(v2)), -1);
        shouldEqual(round(t1 * -fixedPoint(v1) - t2 * fixedPoint(v2)), -1);
        shouldEqual(floor(t1 * -fixedPoint(v4) - t2 * fixedPoint(v8)), -5);
        shouldEqual(ceil(t1 * -fixedPoint(v4) - t2 * fixedPoint(v8)), -5);
        shouldEqual(round(t1 * -fixedPoint(v4) - t2 * fixedPoint(v8)), -5);

        double d1 = 1.0 / 3.0, d2 = 1.0 / 7.0;
        FixedPoint<1, 24> r1(d1), r2(d2);
        FixedPoint<2, 24> r3;
        add(r1, r2, r3);
        shouldEqual(r3.value, (FixedPoint<2, 24>(d1 + d2)).value);
        sub(r1, r2, r3);
        shouldEqual(r3.value, (FixedPoint<2, 24>(d1 - d2)).value);
        mul(r1, r2, r3);
        shouldEqual(r3.value >> 2, (FixedPoint<2, 24>(d1 * d2)).value >> 2);

        for(int i = 0; i < 1024; ++i)
        {
            FixedPoint<4,5> fv1(i, FPNoShift);
            FixedPoint<5,4> fv2(i, FPNoShift);
            FixedPoint<5,5> fv3(i, FPNoShift);
            FixedPoint<6,6> fv4(i, FPNoShift);
            shouldEqual(fixed_point_cast<double>(sqrt(fv1)), floor(sqrt((double)fv1.value)) / 8.0);
            shouldEqual(fixed_point_cast<double>(sqrt(fv2)), floor(sqrt((double)fv2.value)) / 4.0);
            shouldEqual(fixed_point_cast<double>(sqrt(fv3)), floor(sqrt((double)fv3.value)) / 4.0);
            shouldEqual(fixed_point_cast<double>(sqrt(fv4)), floor(sqrt((double)fv4.value)) / 8.0);
        }
    }
};

struct FixedPoint16Test
{
    void testConstruction()
    {
        shouldEqual((FixedPoint16<3>(3).value), 3 << 12);
        shouldEqual((FixedPoint16<3>(-3).value), -3 << 12);
        shouldEqual((-FixedPoint16<3>(3).value), -3 << 12);

        shouldEqual((FixedPoint16<8>(3).value), 3 << 7);

        shouldEqual((FixedPoint16<3>(3.5).value), 7 << 11);
        shouldEqual((FixedPoint16<3>(-3.5).value), -(7 << 11));
        shouldEqual((-FixedPoint16<3>(3.5).value), -(7 << 11));

        shouldEqual((NumericTraits<FixedPoint16<4> >::zero()).value, 0);
        shouldEqual((NumericTraits<FixedPoint16<4> >::one()).value, 1 << 11);
        shouldEqual((NumericTraits<FixedPoint16<4> >::max()).value, (1 << 15) - 1);
        shouldEqual((NumericTraits<FixedPoint16<4> >::min()).value, -(1 << 15));

        shouldEqual((FixedPoint16<1, FPOverflowSaturate>(3.75).value), (1 << 15)-1);
        shouldEqual((FixedPoint16<1, FPOverflowSaturate>(-3.75).value), -(1 << 15));
        try { FixedPoint16<1, FPOverflowError>(3.75); failTest("No exception thrown"); }
        catch(ContractViolation &) {}
        try { FixedPoint16<1, FPOverflowError>(-3.75); failTest("No exception thrown"); }
        catch(ContractViolation &) {}

        FixedPoint16<4> v(3.75);
        shouldEqual((v.value), 15 << 9);
        shouldEqual(((-v).value), -15 << 9);
        shouldEqual((FixedPoint16<4>(v).value), 15 << 9);
        shouldEqual((FixedPoint16<6>(v).value), 15 << 7);
        shouldEqual((FixedPoint16<13>(v).value), 15);
        shouldEqual((FixedPoint16<15>(v).value), 4);

        shouldEqual((FixedPoint16<4>(-v).value), -15 << 9);
        shouldEqual((FixedPoint16<6>(-v).value), -15 << 7);
        shouldEqual((FixedPoint16<13>(-v).value), -15);
        shouldEqual((FixedPoint16<15>(-v).value), -4);

        shouldEqual(fixed_point_cast<double>(v), 3.75);
        shouldEqual(fixed_point_cast<double>(-v), -3.75);
        shouldEqual(frac(v), FixedPoint16<4>(0.75));
        shouldEqual(dual_frac(v), FixedPoint16<4>(0.25));
        shouldEqual(frac(-v), FixedPoint16<4>(0.25));
        shouldEqual(dual_frac(-v), FixedPoint16<4>(0.75));
        shouldEqual(floor(v), 3);
        shouldEqual(ceil(v), 4);
        shouldEqual(floor(-v), -4);
        shouldEqual(ceil(-v), -3);
        shouldEqual(round(v), 4);
        shouldEqual(round(-v), -4);
        shouldEqual(abs(v), v);
        shouldEqual(abs(-v), v);
        shouldEqual(norm(-v), v);
        shouldEqual(squaredNorm(-v), v*v);

        FixedPoint16<2> v1;
        shouldEqual((v1 = v).value, 15 << 11);
        shouldEqual((v1 = -v).value, -15 << 11);

        FixedPoint16<15> v2;
        shouldEqual((v2 = v).value, 4);
        shouldEqual((v2 = -v).value, -4);
    }

    void testComparison()
    {
        FixedPoint16<4> v1(3.75), v2(4);
        FixedPoint16<2> v3(3.75);
        should(v1 == v1);
        should(v1 == v3);
        should(!(v1 != v1));
        should(!(v1 != v3));
        should(v1 <= v1);
        should(v1 <= v3);
        should(!(v1 < v1));
        should(!(v1 < v3));
        should(v1 >= v1);
        should(v1 >= v3);
        should(!(v1 > v1));
        should(!(v1 > v3));

        should(v2 != v1);
        should(v2 != v3);
        should(!(v2 == v1));
        should(!(v2 == v3));
        should(!(v2 <= v1));
        should(!(v2 <= v3));
        should(!(v2 < v1));
        should(!(v2 < v3));
        should(v2 >= v1);
        should(v2 >= v3);
        should(v2 > v1);
        should(v2 > v3);
    }

    void testArithmetic()
    {
        typedef FixedPoint16<1> FP1;
        typedef FixedPoint16<2> FP2;
        typedef FixedPoint16<7> FP7;
        typedef FixedPoint16<8> FP8;
        typedef FixedPoint16<13> FP13;
        typedef FixedPoint16<15> FP15;

        FP1 t0(0), t1(0.75), t2(0.25);
        signed char v1 = 1, v2 = 2, v4 = 4, v8 = 8;

        shouldEqual(FP1(t1) += t1, FP1(1.5));
        shouldEqual(FP1(t1) -= t1, FP1(0.0));
        shouldEqual(FP1(t1) -= t2, FP1(0.5));
        shouldEqual(FP2(t1) *= t1, FP1(9.0 / 16.0));
        shouldEqual(FP2(t1) /= t2, FP2(3));
        shouldEqual(FP2(t1) /= t0, NumericTraits<FP2>::max());
        shouldEqual(FP2(-t1) /= t0, NumericTraits<FP2>::min());

        FP2 res;
        shouldEqual(add(t1, t1, res), FP2(1.5));
        shouldEqual(sub(t1, t1, res), FP2(0));
        shouldEqual(sub(t1, t2, res), FP2(0.5));
        shouldEqual(mul(t1, t1, res), FP2(9.0 / 16.0));
        shouldEqual(div(t1, t2, res), FP2(3));
        shouldEqual(div(t1, t0, res), NumericTraits<FP2>::max());
        shouldEqual(div(-t1, t0, res), NumericTraits<FP2>::min());

        shouldEqual(--t1, FP1(-0.25));
        shouldEqual(t1, FP1(-0.25));
        shouldEqual(++t1, FP1(0.75));
        shouldEqual(t1, FP1(0.75));
        shouldEqual(t1++, FP1(0.75));
        shouldEqual(t1, FP1(1.75));
        shouldEqual(t1--, FP1(1.75));
        shouldEqual(t1, FP1(0.75));

        shouldEqual((t1 * FP7(v1)).value, 3 << 6);
        shouldEqual((t2 * FP7(v1)).value, 1 << 6);
        shouldEqual((-t1 * FP7(v1)).value, -3 << 6);
        shouldEqual((-t2 * FP7(v1)).value, -1 << 6);
        shouldEqual((t1 * -FP7(v1)).value, -3 << 6);
        shouldEqual((t2 * -FP7(v1)).value, -1 << 6);

        shouldEqual((FixedPoint16<2, FPOverflowSaturate>(t1*FP7(v8)).value), (1 << 15)-1);
        shouldEqual((FixedPoint16<2, FPOverflowSaturate>(t1*FP7(-v8)).value), -(1 << 15));
        try { FixedPoint16<2, FPOverflowError>(t1*FP7(v8)); failTest("No exception thrown"); }
        catch(ContractViolation &) {}
        try { FixedPoint16<2, FPOverflowError>(t1*FP7(-v8)); failTest("No exception thrown"); }
        catch(ContractViolation &) {}

        shouldEqual((FP13(t1 * FP7(v1))).value, 3);
        shouldEqual((FP13(t2 * FP7(v1))).value, 1);
        shouldEqual((FP13(-t1 * FP7(v1))).value, -3);
        shouldEqual((FP13(-t2 * FP7(v1))).value, -1);

        shouldEqual((t1 * FP7(v4) + t2 * FP7(v8)).value, 5 << 8);
        shouldEqual((t1 * FP7(v1) + t2 * FP7(v2)).value, 5 << 6);

        shouldEqual(FP7(6) / FP7(3), FP7(2));
        shouldEqual(FP7(0.75) / FP7(0.25), FP7(3));
        shouldEqual(FP7(12) / FP7(48), FP7(0.25));
        shouldEqual(FP1(0.25) / FP7(2), FP7(0.125));
        shouldEqual(FP7(10) / FP1(0.25), FP7(40));
        shouldEqual(FP7(10) / t0, NumericTraits<FP7>::max());
        shouldEqual(FP7(-10) / t0, NumericTraits<FP7>::min());

        shouldEqual(floor(t1 * FP7(v1) + t2 * FP7(v2)), 1);
        shouldEqual(ceil(t1 * FP7(v1) + t2 * FP7(v2)), 2);
        shouldEqual(round(t1 * FP7(v1) + t2 * FP7(v2)), 1);
        shouldEqual(floor(t1 * FP7(v4) + t2 * FP7(v8)), 5);
        shouldEqual(ceil(t1 * FP7(v4) + t2 * FP7(v8)), 5);
        shouldEqual(round(t1 * FP7(v4) + t2 * FP7(v8)), 5);

        shouldEqual(floor(t1 * -FP7(v1) - t2 * FP7(v2)), -2);
        shouldEqual(ceil(t1 * -FP7(v1) - t2 * FP7(v2)), -1);
        shouldEqual(round(t1 * -FP7(v1) - t2 * FP7(v2)), -1);
        shouldEqual(floor(t1 * -FP7(v4) - t2 * FP7(v8)), -5);
        shouldEqual(ceil(t1 * -FP7(v4) - t2 * FP7(v8)), -5);
        shouldEqual(round(t1 * -FP7(v4) - t2 * FP7(v8)), -5);

        double d1 = 1.0 / 3.0, d2 = 1.0 / 7.0;
        FP1 r1(d1), r2(d2);
        FP2 r3;
        add(r1, r2, r3);
        shouldEqual(r3.value, FP2(d1 + d2).value);
        sub(r1, r2, r3);
        shouldEqual(r3.value, FP2(d1 - d2).value);
        mul(r1, r2, r3);
        shouldEqual(r3.value >> 2, FP2(d1 * d2).value >> 2);

        shouldEqual(sqrt(FP7(4)).value, 1 << 12);
        shouldEqual(sqrt(FP8(4)).value, 1 << 12);
        shouldEqual(hypot(FP8(3), FP8(4)), FP8(5));
        shouldEqual(hypot(FP8(-3), FP8(-4)), FP8(5));
        shouldEqual(fixed_point_cast<double>(sqrt(FP7(4))), 2.0);
        shouldEqual(fixed_point_cast<double>(sqrt(FP2(2.25))), 1.5);
        shouldEqual(fixed_point_cast<double>(sqrt(FP8(6.25))), 2.5);

        for(int i = 0; i < 1024; ++i)
        {
            FixedPoint16<11> fv1(i, FPNoShift);
            FixedPoint16<10> fv2(i, FPNoShift);
            FixedPoint16<9>  fv3(i, FPNoShift);
            FixedPoint16<8>  fv4(i, FPNoShift);
            shouldEqual(fixed_point_cast<double>(sqrt(fv1)), floor(sqrt((double)(i << 14))) / 512.0);
            shouldEqual(fixed_point_cast<double>(sqrt(fv2)), floor(sqrt((double)(i << 15))) / 1024.0);
            shouldEqual(fixed_point_cast<double>(sqrt(fv3)), floor(sqrt((double)(i << 14))) / 1024.0);
            shouldEqual(fixed_point_cast<double>(sqrt(fv4)), floor(sqrt((double)(i << 15))) / 2048.0);
        }

        shouldEqual(atan2(FP1(0), FP1(1)), FP2(0));
        shouldEqual(atan2(FP1(0), FP1(-1)), FP2(M_PI));
        shouldEqual(atan2(FP1(1), FP1(0)), FP2(0.5*M_PI));
        shouldEqual(atan2(FP1(-1), FP1(0)), FP2(-0.5*M_PI));

        for(int i = -179; i < 180; ++i)
        {
            double angle = M_PI*i/180.0;
            double c = std::cos(angle), s = std::sin(angle);
            FP2 a = atan2(FP1(s), FP1(c));
            should(abs(i-fixed_point_cast<double>(a)/M_PI*180.0) < 0.3);
            a = atan2(FP15(30000.0*s), FP15(30000.0*c));
            should(abs(i-fixed_point_cast<double>(a)/M_PI*180.0) < 0.3);
        }
    }
};

struct FixedPointTestSuite
: public test_suite
{
    FixedPointTestSuite()
    : test_suite("FixedPointTestSuite")
    {
        add( testCase(&FixedPointTest::testConstruction));
        add( testCase(&FixedPointTest::testComparison));
        add( testCase(&FixedPointTest::testArithmetic));
        add( testCase(&FixedPoint16Test::testConstruction));
        add( testCase(&FixedPoint16Test::testComparison));
        add( testCase(&FixedPoint16Test::testArithmetic));
    }
};

int main(int argc, char ** argv)
{
    FixedPointTestSuite test;

    int failed = test.run(testsToBeExecuted(argc, argv));

    std::cerr << test.report() << std::endl;

    return (failed != 0);
}
