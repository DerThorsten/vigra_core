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
#include <vigra2/array_nd.hxx>
#include <vigra2/convolution_atoms.hxx>

using namespace vigra;

struct ConvolutionAtomsTest
{
    ConvolutionAtomsTest()
    {
    }

    // FIXME: ConvolutionAtomsTest: add more tests

    void convolveLineTest()
    {
        typedef ArrayND<1, double> A;

        A ramp{ 0.0, 1.0, 2.0, 3.0, 4.0 },
          flat(ramp.shape(), 0.0),
          res(ramp.shape(), 0.0);
        int size = ramp.size();

        Kernel1D<double> kernel, akernel;
        kernel.initSymmetricDifference();
        shouldEqual(kernel.borderTreatment(), BORDER_TREATMENT_REFLECT);
        shouldEqual(kernel.symmetry(), KernelOdd);

        akernel.initSymmetricDifference();
        akernel.setAsymmetric();
        shouldEqual(akernel.borderTreatment(), BORDER_TREATMENT_REFLECT);
        shouldEqual(akernel.symmetry(), KernelAsymmetric);

        {
            A expected{ 0.0, 1.0, 1.0, 1.0, 0.0 };
            convolveLine(ramp.data(), size, 0, size, res.data(), kernel);
            should(res == expected);

            convolveLine(flat.data(), size, 0, size, res.data(), kernel);
            should(res == 0.0);

            convolveLine(ramp.data(), size, 0, size, res.data(), akernel);
            should(res == expected);
        }

        {
            A expected{ 0.5, 1.0, 1.0, 1.0, 0.5 };
            convolveLine(ramp.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_REPEAT);
            should(res == expected);

            convolveLine(flat.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_REPEAT);
            should(res == 0.0);

            convolveLine(ramp.data(), size, 0, size, res.data(), akernel, BORDER_TREATMENT_REPEAT);
            should(res == expected);
        }

        {
            res = 1000.0;
            A expected{ 1000.0, 1.0, 1.0, 1.0, 1000.0 };
            convolveLine(ramp.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_AVOID);
            should(res == expected);

            A fexpected{ 1000.0, 0.0, 0.0, 0.0, 1000.0 };
            convolveLine(flat.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_AVOID);
            should(res == fexpected);

            convolveLine(ramp.data(), size, 0, size, res.data(), akernel, BORDER_TREATMENT_AVOID);
            should(res == expected);
        }

        kernel.initBinomial(1);
        akernel.initBinomial(1);
        akernel.setAsymmetric();
        {
            A expected{ 0.25, 1.0, 2.0, 3.0, 2.75 };
            convolveLine(ramp.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_ZEROPAD);
            should(res == expected);

            convolveLine(flat.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_ZEROPAD);
            should(res == 0.0);

            convolveLine(ramp.data(), size, 0, size, res.data(), akernel, BORDER_TREATMENT_ZEROPAD);
            should(res == expected);
        }

        {
            A expected{ 1.25, 1.0, 2.0, 3.0, 2.75 };
            convolveLine(ramp.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_WRAP);
            should(res == expected);

            convolveLine(flat.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_WRAP);
            should(res == 0.0);

            convolveLine(ramp.data(), size, 0, size, res.data(), akernel, BORDER_TREATMENT_WRAP);
            should(res == expected);
        }

        {
            A expected{ 1.0/3.0, 1.0, 2.0, 3.0, 11.0/3.0 };
            convolveLine(ramp.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_CLIP);
            should(res == expected);

            convolveLine(flat.data(), size, 0, size, res.data(), kernel, BORDER_TREATMENT_CLIP);
            should(res == 0.0);

            convolveLine(ramp.data(), size, 0, size, res.data(), akernel, BORDER_TREATMENT_CLIP);
            should(res == expected);
        }
    }

};

struct ConvolutionAtomsTestSuite
: public vigra::test_suite
{
    ConvolutionAtomsTestSuite()
    : vigra::test_suite("ConvolutionAtomsTest")
    {
        add( testCase(&ConvolutionAtomsTest::convolveLineTest));
    }
};

int main(int argc, char ** argv)
{
    ConvolutionAtomsTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
