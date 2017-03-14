/************************************************************************/
/*                                                                      */
/*               Copyright 2016-2017 by Ullrich Koethe                  */
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
#include <vigra2/convolution_kernels.hxx>

struct Kernel1DTest
{
    Kernel1DTest()
    {
    }

    // FIXME: Kernel1DTest: add more tests

    void initExplicitlyTest()
    {
        vigra::Kernel1D<double> k;

        {
            k.initExplicitly(-1,2) = { 1,2,3,4 };

            shouldEqual(k.left(), -1);
            shouldEqual(k.right(), 2);
            shouldEqual(k[-1], 1);
            shouldEqual(k[0], 2);
            shouldEqual(k[1], 3);
            shouldEqual(k[2], 4);
        }

        {
            k.initExplicitly(-1,2) = 1,2,3,4 ;

            shouldEqual(k.left(), -1);
            shouldEqual(k.right(), 2);
            shouldEqual(k[-1], 1);
            shouldEqual(k[0], 2);
            shouldEqual(k[1], 3);
            shouldEqual(k[2], 4);
        }

        {
            k.initExplicitly(-2,1) = -2;
            shouldEqual(k.left(), -2);
            shouldEqual(k.right(), 1);
            shouldEqual(k[-2], -2);
            shouldEqual(k[-1], -2);
            shouldEqual(k[0], -2);
            shouldEqual(k[1], -2);
        }

        try
        {
            k.initExplicitly(-1,1) = 1, 2;
            failTest("no exception thrown");
        }
        catch(vigra::ContractViolation & c)
        {
            std::string expected("\nPrecondition violation!\nKernel1D::initExplicitly(): Wrong number of init values.");
            std::string message(c.what());
            should(0 == expected.compare(message.substr(0,expected.size())));
        }

        try
        {
            k.initExplicitly(-1,1) = 1, 2, 3, 4;
            failTest("no exception thrown");
        }
        catch(vigra::ContractViolation & c)
        {
            std::string expected("\nPrecondition violation!\nKernel1D::initExplicitly(): Wrong number of init values.");
            std::string message(c.what());
            should(0 == expected.compare(message.substr(0,expected.size())));
        }

        try
        {
            k.initExplicitly(-1,1) = { 1, 2, 3, 4 };
            failTest("no exception thrown");
        }
        catch(vigra::ContractViolation & c)
        {
            std::string expected("\nPrecondition violation!\nKernel1D::operator=(std::initializer_list<T>): Wrong number of init values.");
            std::string message(c.what());
            should(0 == expected.compare(message.substr(0,expected.size())));
        }

        try
        {
            k.initExplicitly(1,1) = { 1, 2, 3, 4 };
            failTest("no exception thrown");
        }
        catch(vigra::ContractViolation & c)
        {
            std::string expected("\nPrecondition violation!\nKernel1D::initExplicitly(): left border must be <= 0.");
            std::string message(c.what());
            should(0 == expected.compare(message.substr(0,expected.size())));
        }

        try
        {
            k.initExplicitly(-1,-1) = { 1, 2, 3, 4 };
            failTest("no exception thrown");
        }
        catch(vigra::ContractViolation & c)
        {
            std::string expected("\nPrecondition violation!\nKernel1D::initExplicitly(): right border must be >= 0.");
            std::string message(c.what());
            should(0 == expected.compare(message.substr(0,expected.size())));
        }
    }

};

struct Kernel1DTestSuite
: public vigra::test_suite
{
    Kernel1DTestSuite()
    : vigra::test_suite("Kernel1DTest")
    {
        add( testCase(&Kernel1DTest::initExplicitlyTest));
    }
};

int main(int argc, char ** argv)
{
    Kernel1DTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
