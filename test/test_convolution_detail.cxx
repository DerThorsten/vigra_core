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
#include <vigra2/convolution_detail.hxx>

using namespace vigra;

struct ConvolutionAtomsTest
{
    ConvolutionAtomsTest()
    {
    }

    // FIXME: ConvolutionAtomsTest: add more tests

    void convolveLineTest()
    {
        using namespace vigra::convolution_detail;

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

    void borderCopyTest()
    {
        using namespace vigra::convolution_detail;

        static const int size = 6, kleft = -2, kright = 3,
                         ksize = kright - kleft,
                         outsize = size + ksize;

        int data[size] = {1, 2, 3, 4, 5, 6};
        int out[outsize];

        // copy entire array
        {
            copyLineWithBorderTreatment(data, size, 0, size, out,
                                        kleft, kright, BORDER_TREATMENT_WRAP);
            int r[] = {4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2};
            shouldEqualSequence(out, out+outsize, r);
        }
        {
            copyLineWithBorderTreatment(data, size, 0, size, out,
                                        kleft, kright, BORDER_TREATMENT_REFLECT);
            int r[] = {4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4};
            shouldEqualSequence(out, out+outsize, r);
        }
        {
            copyLineWithBorderTreatment(data, size, 0, size, out,
                                        kleft, kright, BORDER_TREATMENT_REPEAT);
            int r[] = {1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6};
            shouldEqualSequence(out, out+outsize, r);
        }
        {
            copyLineWithBorderTreatment(data, size, 0, size, out,
                                        kleft, kright, BORDER_TREATMENT_AVOID);
            int r[] = {1, 2, 3, 4, 5, 6};
            shouldEqualSequence(out+kright, out+outsize+kleft, r);
        }
        {
            copyLineWithBorderTreatment(data, size, 0, size, out,
                                        kleft, kright, BORDER_TREATMENT_ZEROPAD);
            int r[] = {0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0};
            shouldEqualSequence(out, out+outsize, r);
        }

        // copy with partial border treatment
        {
            copyLineWithBorderTreatment(data, size, 1, size-1, out,
                                        kleft, kright, BORDER_TREATMENT_WRAP);
            int r[] = {5, 6, 1, 2, 3, 4, 5, 6, 1};
            shouldEqualSequence(out, out+outsize-2, r);
        }
        {
            copyLineWithBorderTreatment(data, size, 1, size-1, out,
                                        kleft, kright, BORDER_TREATMENT_REFLECT);
            int r[] = {3, 2, 1, 2, 3, 4, 5, 6, 5};
            shouldEqualSequence(out, out+outsize-2, r);
        }
        {
            copyLineWithBorderTreatment(data, size, 1, size-1, out,
                                        kleft, kright, BORDER_TREATMENT_REPEAT);
            int r[] = {1, 1, 1, 2, 3, 4, 5, 6, 6};
            shouldEqualSequence(out, out+outsize-2, r);
        }
        {
            copyLineWithBorderTreatment(data, size, 1, size-1, out,
                                        kleft, kright, BORDER_TREATMENT_AVOID);
            int r[] = {2, 3, 4, 5};
            shouldEqualSequence(out+kright, out+outsize+kleft-2, r);
        }
        {
            copyLineWithBorderTreatment(data, size, 1, size-1, out,
                                        kleft, kright, BORDER_TREATMENT_ZEROPAD);
            int r[] = {0, 0, 1, 2, 3, 4, 5, 6, 0};
            shouldEqualSequence(out, out+outsize-2, r);
        }

        // copy interior, no border treatment necessary
        {
            copyLineWithBorderTreatment(data, size, kright, size+kleft, out,
                                        kleft, kright, BORDER_TREATMENT_WRAP);
            int r[] = {1, 2, 3, 4, 5, 6};
            shouldEqualSequence(out, out+outsize-ksize, r);
        }
        {
            copyLineWithBorderTreatment(data, size, kright, size+kleft, out,
                                        kleft, kright, BORDER_TREATMENT_REFLECT);
            int r[] = {1, 2, 3, 4, 5, 6};
            shouldEqualSequence(out, out+outsize-ksize, r);
        }
        {
            copyLineWithBorderTreatment(data, size, kright, size+kleft, out,
                                        kleft, kright, BORDER_TREATMENT_REPEAT);
            int r[] = {1, 2, 3, 4, 5, 6};
            shouldEqualSequence(out, out+outsize-ksize, r);
        }
        {
            copyLineWithBorderTreatment(data, size, kright, size+kleft, out,
                                        kleft, kright, BORDER_TREATMENT_AVOID);
            int r[] = {4};
            shouldEqualSequence(out+kright, out+outsize+kleft-ksize, r);
        }
        {
            copyLineWithBorderTreatment(data, size, kright, size+kleft, out,
                                        kleft, kright, BORDER_TREATMENT_ZEROPAD);
            int r[] = {1, 2, 3, 4, 5, 6};
            shouldEqualSequence(out, out+outsize-ksize, r);
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
        add( testCase(&ConvolutionAtomsTest::borderCopyTest));
    }
};

int main(int argc, char ** argv)
{
    ConvolutionAtomsTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
