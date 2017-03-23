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

#pragma once

#ifndef VIGRA2_CONVOLUTION_DETAIL_HXX
#define VIGRA2_CONVOLUTION_DETAIL_HXX

#include "numeric_traits.hxx"
#include "mathutil.hxx"
#include "convolution_kernels.hxx"

namespace vigra {

/** \addtogroup SeparableConvolution One-dimensional and separable convolution functionality
*/
//@{

namespace convolution_detail {

template <KernelSymmetry SYMMETRY = KernelEven>
struct AddBySymmetry
{
    template <class T>
    T operator()(T a, T b) const
    {
        return a + b;
    }
};

template <>
struct AddBySymmetry<KernelOdd>
{
    template <class T>
    T operator()(T a, T b) const
    {
        return a - b;
    }
};

template <ArrayIndex RADIUS>
struct RadiusLoopUnrolling
{
    constexpr ArrayIndex operator()(ArrayIndex) const
    {
        return RADIUS;
    }
};

template <>
struct RadiusLoopUnrolling<runtime_size>
{
    ArrayIndex operator()(ArrayIndex radius) const
    {
        return radius;
    }
};

template <ArrayIndex RADIUS = runtime_size, KernelSymmetry SYMMETRY = KernelAsymmetric>
struct ConvolveBorder
{
        // Implementation for KernelEven and KernelOdd
        //
        // 'out' must be the point corresponding to 'in+start'.
        // 'kernel' must point to the kernel's center point, so that
        // 'kernel[left]' (with left <= 0) and 'kernel[right]' (with
        //  right >= 0) are both valid.
    template <class T1, class T2, class T3>
    static void exec_x(T1 * in, ArrayIndex size, ArrayIndex start, ArrayIndex end,
                       T2 * out,
                       T3 * kernel, ArrayIndex left, ArrayIndex right,
                       BorderTreatmentMode border)
    {
        typedef PromoteType<T1, T3> SumType;

        AddBySymmetry<SYMMETRY> addsub;
        RadiusLoopUnrolling<RADIUS> const_radius;

        vigra_assert(-left == right,
            "convolve(): even and odd filters must have equal left and right window size.");
        ArrayIndex radius = right;

        switch(border)
        {
          case BORDER_TREATMENT_CLIP:
          {
            vigra_precondition(SYMMETRY == KernelEven,
                "convolve(): BORDER_TREATMENT_CLIP only makes sense for even kernels.");
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum  = kernel[0] * in[x],
                        ksum = kernel[0];

                for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
                {
                    ArrayIndex l = x - k;
                    if(l < 0)
                        break;

                    sum  += kernel[k] * in[l];
                    ksum += kernel[k];
                }

                for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
                {
                    ArrayIndex r = x + k;
                    if(r >= size)
                        break;

                    sum  += kernel[k] * in[r];
                    ksum += kernel[k];
                }

                if(ksum != SumType())
                    *out = sum / ksum;
            }
            break;
          }
          case BORDER_TREATMENT_REPEAT:
          {
            ArrayIndex s = size - 1;
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum = kernel[0] * in[x];

                for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
                {
                    ArrayIndex l = x - k,
                               r = x + k;
                    if(l < 0)
                        l = 0;
                    if(r > s)
                        r = s;

                    sum += kernel[k] * addsub(in[l], in[r]);
                }

                *out = sum;
            }
            break;
          }
          case BORDER_TREATMENT_REFLECT:
          {
            ArrayIndex s = 2*size - 2;
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum = kernel[0] * in[x];

                for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
                {
                    ArrayIndex l = abs(x - k),
                               r = x + k;
                    if(r >= size)
                        r = s - r;

                    sum += kernel[k] * addsub(in[l], in[r]);
                }

                *out = sum;
            }
            break;
          }
          case BORDER_TREATMENT_WRAP:
          {
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum = kernel[0] * in[x];

                for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
                {
                    ArrayIndex l = x - k,
                               r = x + k;
                    if(l < 0)
                        l += size;
                    if(r >= size)
                        r -= size;

                    sum += kernel[k] * addsub(in[l], in[r]);
                }

                *out = sum;
            }
            break;
          }
          case BORDER_TREATMENT_ZEROPAD:
          {
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum = kernel[0] * in[x];

                for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
                {
                    ArrayIndex l = x - k;
                    if(l < 0)
                        break;

                    sum += kernel[k] * in[l];
                }

                for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
                {
                    ArrayIndex r = x + k;
                    if(r >= size)
                        break;

                    if(SYMMETRY == KernelEven)
                        sum += kernel[k] * in[r];
                    else
                        sum -= kernel[k] * in[r];
                }

                *out = sum;
            }
            break;
          }
        }
    }
};

template <ArrayIndex RADIUS>
struct ConvolveBorder<RADIUS, KernelAsymmetric>
{
        // Implementation for KernelAsymmetric
        //
        // 'out' must be the point corresponding to 'in+start'.
        // 'kernel' must point to the kernel's center point, so that
        // 'kernel[left]' (with left <= 0) and 'kernel[right]' (with
        //  right >= 0) are both valid.
    template <class T1, class T2, class T3>
    static void exec_x(T1 * in, ArrayIndex size, ArrayIndex start, ArrayIndex end,
                       T2 * out,
                       T3 * kernel, ArrayIndex left, ArrayIndex right,
                       BorderTreatmentMode border)
    {
        typedef PromoteType<T1, T3> SumType;

        switch(border)
        {
          case BORDER_TREATMENT_CLIP:
          {
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum  = kernel[0] * in[x],
                        ksum = kernel[0];

                // note that the kernel must be reflected
                for (ArrayIndex k = -1; k >= left; --k)
                {
                    ArrayIndex r = x - k;
                    if(r >= size)
                        break;

                    sum  += kernel[k] * in[r];
                    ksum += kernel[k];
                }

                for (ArrayIndex k = 1; k <= right; ++k)
                {
                    ArrayIndex l = x - k;
                    if(l < 0)
                        break;

                    sum  += kernel[k] * in[l];
                    ksum += kernel[k];
                }

                if(ksum != SumType())
                    *out = sum / ksum;
            }
            break;
          }
          case BORDER_TREATMENT_REPEAT:
          {
            ArrayIndex s = size - 1;
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum  = kernel[0] * in[x];

                // note that the kernel must be reflected
                for (ArrayIndex k = -1; k >= left; --k)
                {
                    ArrayIndex r = x - k;
                    if(r >= size)
                        r = s;

                    sum += kernel[k] * in[r];
                }

                for (ArrayIndex k = 1; k <= right; ++k)
                {
                    ArrayIndex l = x - k;
                    if(l < 0)
                        l = 0;

                    sum += kernel[k] * in[l];
                }

                *out = sum;
            }
            break;
          }
          case BORDER_TREATMENT_REFLECT:
          {
            ArrayIndex s = 2*size - 2;
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum  = kernel[0] * in[x];

                // note that the kernel must be reflected
                for (ArrayIndex k = -1; k >= left; --k)
                {
                    ArrayIndex r = x - k;
                    if(r >= size)
                        r = s - r;

                    sum += kernel[k] * in[r];
                }

                for (ArrayIndex k = 1; k <= right; ++k)
                {
                    ArrayIndex l = abs(x - k);

                    sum += kernel[k] * in[l];
                }

                *out = sum;
            }
            break;
          }
          case BORDER_TREATMENT_WRAP:
          {
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum  = kernel[0] * in[x];

                // note that the kernel must be reflected
                for (ArrayIndex k = -1; k >= left; --k)
                {
                    ArrayIndex r = x - k;
                    if(r >= size)
                        r -= size;

                    sum += kernel[k] * in[r];
                }

                for (ArrayIndex k = 1; k <= right; ++k)
                {
                    ArrayIndex l = x - k;
                    if(l < 0)
                        l += size;

                    sum += kernel[k] * in[l];
                }

                *out = sum;
            }
            break;
          }
          case BORDER_TREATMENT_ZEROPAD:
          {
            for(ArrayIndex x = start; x < end; ++x, ++out)
            {
                SumType sum  = kernel[0] * in[x];

                // note that the kernel must be reflected
                for (ArrayIndex k = -1; k >= left; --k)
                {
                    ArrayIndex r = x - k;
                    if(r >= size)
                        break;

                    sum += kernel[k] * in[r];
                }

                for (ArrayIndex k = 1; k <= right; ++k)
                {
                    ArrayIndex l = x - k;
                    if(l < 0)
                        break;

                    sum += kernel[k] * in[l];
                }

                *out = sum;
            }
            break;
          }
        }
    }
};

template <ArrayIndex RADIUS = runtime_size, KernelSymmetry SYMMETRY = KernelAsymmetric>
struct ConvolveLine
{
        // Implementation for KernelEven and KernelOdd
    template <class T1, class T2, class T3>
    static void exec_x(T1 * in, ArrayIndex size, ArrayIndex start, ArrayIndex end,
                       T2 * out,
                       T3 * kernel, ArrayIndex left, ArrayIndex right,
                       BorderTreatmentMode border)
    {
        typedef PromoteType<T1, T3> SumType;
        typedef ConvolveBorder<RADIUS, SYMMETRY> Border;

        AddBySymmetry<SYMMETRY> addsub;
        RadiusLoopUnrolling<RADIUS> const_radius;

        vigra_assert(-left == right,
            "convolve(): even and odd filters must have equal left and right window size.");
        ArrayIndex radius = right;

        vigra_precondition(size >= radius + 1,
            "convolve(): kernel radius exceeds array size. "
            "Consider decreasing std_dev or window_ratio.");

        if(2*radius >= size)
        {
            // all points need border treatment
            Border::exec_x(in, size, start, end, out, kernel, left, right, border);
        }
        else
        {
            // border treatment bounds
            ArrayIndex left_bound  = max(0, radius - start),
                       inner_start = start + left_bound,
                       right_bound = max(0, radius - (size - end)),
                       inner_end   = end - right_bound;

            // left border treatment
            Border::exec_x(in, size, start, inner_start, out,
                           kernel, left, right, border);
            out += left_bound;

            // no border treatment
            for(ArrayIndex x = inner_start; x < inner_end; ++x, ++out)
            {
                SumType sum = kernel[0] * in[x];

                for (ArrayIndex k = 1; k <= const_radius(radius); ++k)
                {
                    sum += kernel[k] * addsub(in[x - k], in[x + k]);
                }

                *out = sum;
            }

            // right border treatment
            Border::exec_x(in, size, inner_end, end, out,
                           kernel, left, right, border);
        }
    }
};

template <ArrayIndex RADIUS>
struct ConvolveLine<RADIUS, KernelAsymmetric>
{
        // Implementation for KernelAsymmetric
    template <class T1, class T2, class T3>
    static void exec_x(T1 * in, ArrayIndex size, ArrayIndex start, ArrayIndex end,
                       T2 * out,
                       T3 * kernel, ArrayIndex left, ArrayIndex right,
                       BorderTreatmentMode border)
    {
        typedef PromoteType<T1, T3> SumType;
        typedef ConvolveBorder<RADIUS, KernelAsymmetric> Border;

        vigra_precondition(size >= max(right, -left) + 1,
            "convolve(): kernel radius exceeds array size.");

        if(right - left >= size)
        {
            // all points need border treatment
            Border::exec_x(in, size, start, end, out, kernel, left, right, border);
        }
        else
        {
            // border treatment bounds
            // (note that the kernel must be reflected)
            ArrayIndex left_bound  = max(0, right - start),
                       inner_start = start + left_bound,
                       right_bound = max(0, -left - (size - end)),
                       inner_end   = end - right_bound;

            // left border treatment
            Border::exec_x(in, size, start, inner_start, out,
                           kernel, left, right, border);
            out += left_bound;

            // no border treatment
            for(ArrayIndex x = inner_start; x < inner_end; ++x, ++out)
            {
                SumType sum = SumType();

                for (ArrayIndex k = left; k <= right; ++k)
                {
                    sum += kernel[k] * in[x - k];
                }

                *out = sum;
            }

            // right border treatment
            Border::exec_x(in, size, inner_end, end, out,
                           kernel, left, right, border);
        }
    }
};

template <ArrayIndex RADIUS, class T1, class T2, class T3>
void convolveDispatch(T1 * in, ArrayIndex size, ArrayIndex start, ArrayIndex end,
                      T2 * out,
                      Kernel1D<T3> const & kernel, BorderTreatmentMode border)
{
    if(border == BORDER_TREATMENT_DEFAULT)
        border = kernel.borderTreatment();
    if(border == BORDER_TREATMENT_DEFAULT)
        border = BORDER_TREATMENT_REFLECT;

    // enable optimized processing of even and odd kernels
    switch(kernel.symmetry())
    {
      case KernelEven:
      {
        ConvolveLine<RADIUS, KernelEven>::exec_x(in, size, start, end, out,
                                                 kernel.center(), kernel.left(), kernel.right(),
                                                 border);
        break;
      }
      case KernelOdd:
      {
        ConvolveLine<RADIUS, KernelOdd>::exec_x(in, size, start, end, out,
                                                kernel.center(), kernel.left(), kernel.right(),
                                                border);
        break;
      }
      default:
      {
        ConvolveLine<RADIUS>::exec_x(in, size, start, end, out,
                                     kernel.center(), kernel.left(), kernel.right(),
                                     border);
      }
    }
}

template <class T1, class T2, class T3>
void convolveLine(T1 * in, ArrayIndex size, ArrayIndex start, ArrayIndex end,
                  T2 * out,
                  Kernel1D<T3> const & kernel,
                  BorderTreatmentMode border = BORDER_TREATMENT_DEFAULT)
{
    ArrayIndex radius = (kernel.right() == -kernel.left())
                            ? kernel.right()
                            : runtime_size;

    // enable loop unrolling for small kernels
    switch(radius)
    {
      case 0:
      {
        convolveDispatch<0>(in, size, start, end, out, kernel, border);
        break;
      }
      case 1:
      {
        convolveDispatch<1>(in, size, start, end, out, kernel, border);
        break;
      }
      case 2:
      {
        convolveDispatch<2>(in, size, start, end, out, kernel, border);
        break;
      }
      case 3:
      {
        convolveDispatch<3>(in, size, start, end, out, kernel, border);
        break;
      }
      case 4:
      {
        convolveDispatch<4>(in, size, start, end, out, kernel, border);
        break;
      }
      case 5:
      {
        convolveDispatch<5>(in, size, start, end, out, kernel, border);
        break;
      }
      case 6:
      {
        convolveDispatch<6>(in, size, start, end, out, kernel, border);
        break;
      }
      case 7:
      {
        convolveDispatch<7>(in, size, start, end, out, kernel, border);
        break;
      }
      case 8:
      {
        convolveDispatch<8>(in, size, start, end, out, kernel, border);
        break;
      }
      case 9:
      {
        convolveDispatch<9>(in, size, start, end, out, kernel, border);
        break;
      }
      default:
      {
        convolveDispatch<runtime_size>(in, size, start, end, out, kernel, border);
      }
    }
}

} // namespace convolution_detail

//@}

} // namespace vigra

#endif /* VIGRA2_CONVOLUTION_DETAIL_HXX */
