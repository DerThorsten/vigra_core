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
#include <vigra2/mathutil.hxx>

using namespace vigra;

struct MathUtilTest
{
    void testSpecialIntegerFunctions()
    {
        for(int32_t i = 0; i < 1024; ++i)
        {
            shouldEqual(sqrti(i), (int32_t)floor(sqrt((double)i)));
        }

        shouldEqual(roundi(0.0), 0);
        shouldEqual(roundi(1.0), 1);
        shouldEqual(roundi(1.1), 1);
        shouldEqual(roundi(1.6), 2);
        shouldEqual(roundi(-1.0), -1);
        shouldEqual(roundi(-1.1), -1);
        shouldEqual(roundi(-1.6), -2);

        uint32_t roundPower2[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 0xffff, 0x7fffffff, 0x80000000, 0x80000001, 0xffffffff};
        uint32_t floorResult[] = {0, 1, 2, 2, 4, 4, 4, 8, 8, 8, 16, 0x8000, 0x40000000, 0x80000000, 0x80000000, 0x80000000};
        uint32_t ceilResult[] = {0, 1, 2, 4, 4, 8, 8, 8, 16, 16, 16, 0x10000, 0x80000000, 0x80000000, 0, 0};
        for(unsigned int i = 0; i < sizeof(roundPower2) / sizeof(uint32_t); ++i)
        {
            shouldEqual(floorPower2(roundPower2[i]), floorResult[i]);
            shouldEqual(ceilPower2(roundPower2[i]), ceilResult[i]);
        }

        for(int32_t k=0; k<32; ++k)
        {
            shouldEqual(log2i(1 << k), k);
            shouldEqual(log2i((1 << k) + 1), k == 0 ? 1 : k);
            shouldEqual(log2i((1 << k) - 1), k-1);
        }

        should(even(0));
        should(!odd(0));
        should(!even(1));
        should(odd(1));
        should(even(2));
        should(!odd(2));
        should(!even(-1));
        should(odd(-1));
        should(even(-2));
        should(!odd(-2));
    }


    void testSpecialFunctions()
    {
        shouldEqualTolerance(erf(0.3), 0.32862675945912745, 1e-7);

        for(double x = -4.0; x <= 4.0; x += 1.0)
        {
            shouldEqual(sin_pi(x), 0.0);
            shouldEqual(cos_pi(x+0.5), 0.0);
        }

        for(double x = -4.5; x <= 4.5; x += 2.0)
        {
            shouldEqual(sin_pi(x), -1.0);
            shouldEqual(cos_pi(x+0.5), 1.0);
        }

        for(double x = -3.5; x <= 4.5; x += 2.0)
        {
            shouldEqual(sin_pi(x), 1.0);
            shouldEqual(cos_pi(x+0.5), -1.0);
        }

        for(double x = -4.0; x <= 4.0; x += 0.0625)
        {
            shouldEqualTolerance(sin_pi(x), std::sin(M_PI*x), 1e-14);
            shouldEqualTolerance(cos_pi(x), std::cos(M_PI*x), 1e-14);
        }

        shouldEqualTolerance(sin_pi(0.25), 0.5*M_SQRT2, 2e-16);
        shouldEqualTolerance(cos_pi(0.25), 0.5*M_SQRT2, 2e-16);

        // NOTE: the vigra:: specification is needed due to a possible GCC 6.x bug (gamma() function present
        // in global namespace).
        shouldEqual(vigra::gamma(4.0), 6.0);
        shouldEqualTolerance(vigra::gamma(0.1), 9.5135076986687306, 1e-15);
        shouldEqualTolerance(vigra::gamma(3.2), 2.4239654799353683, 1e-15);
        shouldEqualTolerance(vigra::gamma(170.2), 1.1918411166366696e+305, 1e-15);
        shouldEqualTolerance(vigra::gamma(-0.1), -10.686287021193193, 1e-14);
        shouldEqualTolerance(vigra::gamma(-3.2), 0.689056412005979, 1e-14);
        shouldEqualTolerance(vigra::gamma(-170.2), -2.6348340538196879e-307, 1e-14);
        try { vigra::gamma(0.0); failTest("No exception thrown"); } catch(ContractViolation &) {}
        try { vigra::gamma(-1.0); failTest("No exception thrown"); } catch(ContractViolation &) {}

        shouldEqual(loggamma(1.0), 0.0);
        shouldEqual(loggamma(2.0), 0.0);
        shouldEqualTolerance(loggamma(4.0e-22), 49.2705776847491144296, 1e-15);
        shouldEqualTolerance(loggamma(0.1), 2.2527126517342055401, 1e-15);
        shouldEqualTolerance(loggamma(0.3), 1.0957979948180756047, 1e-15);
        shouldEqualTolerance(loggamma(0.8), 0.15205967839983755563, 1e-15);
        shouldEqualTolerance(loggamma(1.1), -0.049872441259839757344, 1e-15);
        shouldEqualTolerance(loggamma(1.3), -0.10817480950786048655, 1e-15);
        shouldEqualTolerance(loggamma(1.8), -0.071083872914372153717, 1e-15);
        shouldEqualTolerance(loggamma(3.0), 0.69314718055994528623, 1e-15);
        shouldEqualTolerance(loggamma(3.1), 0.78737508327386251938, 1e-15);
        shouldEqualTolerance(loggamma(4.0), 1.79175946922805500081, 1e-15);
        shouldEqualTolerance(loggamma(8.0), 8.5251613610654143002, 1e-15);
        shouldEqualTolerance(loggamma(1000.0), 5905.2204232091812118261, 1e-15);
        shouldEqualTolerance(loggamma(1000.2), 5906.6018942569799037, 1e-15);
        shouldEqualTolerance(loggamma(2.8e+17), 1.096859847946237952e+19, 1e-15);
        shouldEqualTolerance(loggamma(2.9e+17), 1.1370510622188449792e+19, 1e-15);
        shouldEqualTolerance(loggamma(5.7646075230342349e+17), 2.2998295812288974848e+19, 1e-15);
        try { loggamma(0.0); failTest("No exception thrown"); } catch(ContractViolation &) {}
        try { loggamma(-1.0); failTest("No exception thrown"); } catch(ContractViolation &) {}
    }
};
struct MathUtilTestSuite
: public test_suite
{
    MathUtilTestSuite()
    : test_suite("MathUtilTestSuite")
    {
        add( testCase(&MathUtilTest::testSpecialIntegerFunctions));
        add( testCase(&MathUtilTest::testSpecialFunctions));
    }
};

int main(int argc, char ** argv)
{
  try
  {
    MathUtilTestSuite test;

    int failed = test.run(testsToBeExecuted(argc, argv));

    std::cerr << test.report() << std::endl;

    return (failed != 0);
  }
  catch(std::exception & e)
  {
    std::cerr << "Unexpected exception: " << e.what() << "\n";
    return 1;
  }
}
