#ifndef WATERCOLOR_HPP
#define WATERCOLOR_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace pcv
{
namespace math
{
static double PI = std::acos(-1);
// im2col with HWC data structure
// 1 dilation, 1 stride, legacy `same` for padding
// @see caffe2
template<typename T>
T *im2col(const T* const &im,
          int const &im_height,
          int const &im_width,
          int const &im_channel,
          int const &im_data_stride,
          int const &kernel_h,
          int const &kernel_w,
          int pad_h,
          int const &pad_w)
{
    auto col = new T[im_height*im_width*kernel_h*kernel_w*im_channel];
    pad_h *= -1;
    auto src = 0;
    auto channel_size = sizeof(T) * im_channel;
    for (auto h = 0; h < im_height; ++h)
    {
        auto w_pad = -pad_w;
        for (auto w = 0; w < im_width; ++w)
        {
            for (auto y = pad_h; y < pad_h + kernel_h; ++y)
            {
                for (auto x = w_pad; x < w_pad + kernel_w; ++x)
                {
                    if (y > -1 && y < im_height && x > -1 && x < im_width)
                    {
                        memcpy(col+src,
                               im + (y * im_width + x) * im_data_stride,
                               channel_size);
                    } else {
                        memset(col+src,
                               0,
                               channel_size);
                    }
                    src += im_channel;
                }
            }
            ++w_pad;
        }
        ++pad_h;
    }
    return col;
} // pcv::im2col

// convolution with HWC data structure
// 1 dilation, 1 stride, legacy `same` for padding
template<typename T>
double * conv2(const T *const &im,
               int const &im_height,
               int const &im_width,
               int const &im_channel,
               int const &im_data_stride,
               const double * const &kernel,
               int const &kernel_h,
               int const &kernel_w)
{
    assert(kernel_h % 2 == 1);
    assert(kernel_w % 2 == 1);

    auto im_size = im_height * im_width;
    auto new_im = new double[im_size * im_channel];
    auto kernel_vol = kernel_h * kernel_w;
    auto pad_h = kernel_h / 2;
    auto pad_w = kernel_w / 2;
    auto col = im2col(im,
                      im_height, im_width, im_channel, im_data_stride,
                      kernel_h, kernel_w,
                      pad_h, pad_w);
    auto src = 0;
    for (auto i= 0; i < im_size; ++i)
    {
        auto tar = i*im_channel*kernel_vol;
        for (auto c = 0; c < im_channel; ++c, ++src, ++tar)
        {
            auto tar0 = 0;
            new_im[src] = 0;
            for (auto k = 0; k < kernel_vol; ++k, tar0+=im_channel)
                new_im[src] += col[tar+tar0] * kernel[k];
        }
    }
    delete [] col;
    return new_im;
} // pcv::conv2

} // pcv::math

namespace serialize
{
std::string px(const unsigned char *const &data,
               std::uint32_t const &w,
               std::uint32_t const &h,
               std::uint32_t const &comp)
{
    std::string res("PXIMG");
    std::uint32_t dim[] = {
            1, // version
            h, w, comp
                           };
    res.append(reinterpret_cast<char *>(dim), sizeof(dim));

    std::uint64_t tot = w*h;
    std::uint8_t raw_data[(tot+2)*comp];
    auto img_data = raw_data + 2*comp;
    auto tar = -1;

    for (std::uint32_t ch = 0; ch < comp; ++ch)
    {
        auto thresh = raw_data + ch*2;
        thresh[0] = 0xFF;
        thresh[1] = 0x00;
        for (std::uint64_t i = 0; i < tot; ++i)
        {
            auto s = i*comp + ch;
            if (data[s] < thresh[0])
                thresh[0] = data[s];
            if (data[s] > thresh[1])
                thresh[1] = data[s];
            if (thresh[0] == 0x00 && thresh[1] == 0xFF)
                break;
        }
        if (thresh[1] == thresh[0])
            continue;
        auto interval = thresh[1] - thresh[0] > 15 ? (thresh[1] - thresh[0])/15 : 1;
        auto last_val = -1;
        auto count = -1;
        for (std::uint64_t i = 0; i < tot; ++i)
        {
            auto src = i*comp + ch;
            auto val = 0x0F & ((data[src+comp]-thresh[0])/interval);
            if (val == last_val)
            {
                if (count == 15)
                {
                    img_data[++tar] = 0xF0 + last_val;
                    count = 0;
                }
                else
                    ++count;
            }
            else
            {
                if (count != -1)
                    img_data[++tar] = (count << 4) + last_val;
                count = 0;
                last_val = val;
            }
        }
        img_data[++tar] = (count << 4) + last_val;
    }

    res.append(reinterpret_cast<char *>(raw_data), sizeof(std::uint8_t)*(2*comp+1+tar));

    return res;
}
} // pcv::serialize

namespace deserialize
{
template<typename N0, typename N1, typename N2,
        typename Num>
std::uint8_t *px(std::string const &data, // better to use string_view
                 N0 &w,
                 N1 &h,
                 N2 &channels_in_file,
                 Num const &desired_channels)
{
    if (data.at(0) != 'P' || data.at(1) != 'X' ||
        data.at(2) != 'I' || data.at(3) != 'M' || data.at(4) != 'G')
        throw std::invalid_argument("Invalid signature of PXIMG format");
    auto data_info = reinterpret_cast<const std::uint32_t *>(
            reinterpret_cast<const char *>(data.data()) + 5
    );
    // no version check so far
    h = data_info[1];
    w = data_info[2];
    channels_in_file = data_info[3];
    auto ch = static_cast<std::uint64_t>(desired_channels);
    if (data_info[3] > ch)
        throw std::invalid_argument("Fail to pass channel number check");

    auto img_data = reinterpret_cast<const std::uint8_t *>(data_info + 4);
    auto res = new std::uint8_t[data_info[1] * data_info[2] * ch];

    std::uint64_t tot = h*w, tar = 2*data_info[3];
    for (std::uint32_t c = 0; c < data_info[3]; ++c)
    {
        auto min = img_data[c*2];
        auto max = img_data[c*2+1];
        if (max == min)
        {
            for (std::uint64_t i = 0; i < tot; ++i)
                res[i*ch + c] = min;
            continue;
        }
        auto interval = img_data[c*2+1] - min > 15 ? (img_data[c*2+1] - min)/15 : 1;
        for (std::uint64_t i = 0; i < tot;)
        {
            auto t = img_data[tar] >> 4;
            auto val = (0x0F & img_data[tar])*interval;
            for (auto t0 = 0; t0 < t+1; ++t0, ++i)
                res[i * ch + c] = min + val;
            ++tar;
        }
    }

    return res;
};
} // pcv::deserialize

namespace write
{
void px(std::string const &filename, // better to use string_view
        std::uint64_t const &w,
        std::uint64_t const &h,
        std::uint64_t const &comp,
        const unsigned char *const &data)
{
    std::ofstream f(filename, std::ios::out | std::ios::binary);
    if (f.is_open())
    {
        try
        {
            f << serialize::px(data, w, h, comp);
        }
        catch (...)
        {
            f.close();
            std::rethrow_exception(std::current_exception());
        }
    }
    else
        throw std::invalid_argument("Failed to open file " + filename);
    f.close();
}
} // pcv::write

namespace load
{
template<typename N0, typename N1, typename N2,
         typename Num>
std::uint8_t *px(std::string const &filename, // better to use string_view
                 N0 &w,
                 N1 &h,
                 N2 &channels,
                 Num const &desired_channels)
{
    std::uint8_t *res;
    std::ifstream f(filename, std::ios::in | std::ios::binary);
    if (f.is_open())
    {
        try
        {
            res = deserialize::px(std::string(std::istreambuf_iterator<char>(f),
                                              std::istreambuf_iterator<char>()),
                                  w, h, channels, desired_channels);
        }
        catch (...)
        {
            f.close();
            std::rethrow_exception(std::current_exception());
        }
    }
    else
        throw std::invalid_argument("Failed to open file " + filename);
    f.close();
    return res;
}
} // pcv::load

namespace conversion
// all conversion in the bound [0, 255], no bound check for each function
{
template<typename T_IN, typename T_OUT>
void rgb2gray(T_IN const &R, T_IN const &G, T_IN const &B,
              T_OUT &gray)
{
    gray = 0.299*R + 0.587*G + 0.114*B;
};
template<typename T_IN, typename T_OUT>
void rgb2hsl(T_IN const &R, T_IN const &G, T_IN const &B,
             T_OUT &H, T_OUT &S, T_OUT &L)
{
    auto min = std::min(std::min(R, G), B);
    auto max = std::max(std::max(R, G), B);

    if (max == 0)
    {
        H = 0;
        S = 0;
        L = 0;
        return;
    }
    else if (max == min)
    {
        H = 0;
        S = 0;
    }
    else
    {
        double h;
        if (max == R)
            h = static_cast<double>(G-B)/(max-min);
        else if (max == G)
            h = static_cast<double>(B-R)/(max-min) + 2;
        else // if (max == B)
            h = static_cast<double>(R-G)/(max-min) + 4;
        if (h < 0)
            h += 6;
        H = h/6*255;
        S = 1.0*(max-min) / (max+min > 127 ? 510 - max - min : max + min) * 255;
    }
    L = 0.5*(max+min);
};
template<typename T_IN, typename T_OUT>
void hsl2rgb(T_IN const &H, T_IN const &S, T_IN const &L,
             T_OUT &R, T_OUT &G, T_OUT &B)
{
    auto l = L/255.f;
    auto s = S/255.f;
    auto q = l < 0.5 ? l*(1+s) : l + s - l*s;
    auto p = 2*l-q;
    auto tg = H/255.f;
    auto tr = 1.f/3.f;
    auto tb = tg - tr;
    tr += tg;
    if (tr > 1)
        tr -= 1;
    if (tb < 0)
        tb += 1;
    if (tr < 1.f/6.f)
        R = 255*(p+((q-p)*6*tr));
    else if (tr < 0.5f)
        R = 255*q;
    else if (tr < 2.f/3.f)
        R = 255*(p+((q-p)*6*(2.f/3.f-tr)));
    else
        R = 255*p;
    if (tg < 1.f/6.f)
        G = 255*(p+((q-p)*6*tg));
    else if (tg < 0.5f)
        G = 255*q;
    else if (tg < 2.f/3.f)
        G = 255*(p+((q-p)*6*(2.f/3.f-tg)));
    else
        G = 255*p;
    if (tb < 1.f/6.f)
        B = 255*(p+((q-p)*6*tb));
    else if (tb < 0.5f)
        B = 255*q;
    else if (tb < 2.f/3.f)
        B = 255*(p+((q-p)*6*(2.f/3.f-tb)));
    else
        B = 255*p;
};
template<typename T_IN, typename T_OUT>
void rgb2hsv(T_IN const &R, T_IN const &G, T_IN const &B,
             T_OUT &H, T_OUT &S, T_OUT &V)
{
        auto min = R < G ? (R < B ? R : B) : (G > B ? B : G);
        auto max = R > G ? (R > B ? R : B) : (G < B ? B : G);
        if (max == 0 || max == min)
        {
            H = 0;
            S = 0;
            V = max;
            return;
        }
        V = max;
        S = 255*static_cast<double>(max-min) / max;
        if (max == R)
            H = (static_cast<double>(G-B)/(max - min)) / 6*255;
        else if (max == G)
            H = (2 + static_cast<double>(B-R)/(max - min)) / 6*255;
        else // if (max == B)
            H = (4 + static_cast<double>(R-G)/(max - min)) / 6*255;
        if (H < 0)
            H += 255;
};
template<typename T_IN, typename T_OUT>
void hsv2rgb(T_IN const &H, T_IN const &S, T_IN const &V,
             T_OUT &R, T_OUT &G, T_OUT &B)
{
    auto h = H*6.0/255;
    auto s = S/255.0;
    double i, f;
    f = std::modf(h, &i);
    switch (static_cast<int>(i))
    {
        case 0:
            R = V;
            G = V * (1 - s * (1- f));
            B = V * (1 - s);
            break;
        case 1:
            R = V * (1 - s * f);
            G = V;
            B = V * (1 - s);
            break;
        case 2:
            R = V * (1 - s);
            G = V;
            B = V * (1 - s * (1- f));
            break;
        case 3:
            R = V * (1 - s);
            G = V * (1 - s * f);
            B = V;
            break;
        case 4:
            R = V * (1 - s * (1- f));
            G = V * (1 - s);
            B = V;
            break;
        default: // case 5:
            R = V;
            G = V * (1 - s);
            B = V * (1 - s * f);
            break;
    }
};
template<typename T_IN, typename T_OUT>
void rgb2xyz(T_IN const &R, T_IN const &G, T_IN const &B,
             T_OUT &X, T_OUT &Y, T_OUT &Z)
{
    auto dr = static_cast<float>(R)/255;
    auto dg = static_cast<float>(G)/255;
    auto db = static_cast<float>(B)/255;

    if (dr > 0.04045)
        dr = std::pow((dr+0.055)/1.055, 2.4);
    else
        dr /= 12.92;
    if (dg > 0.04045)
        dg = std::pow((dg+0.055)/1.055, 2.4);
    else
        dg /= 12.92;
    if (db > 0.04045)
        db = std::pow((db+0.055)/1.055, 2.4);
    else
        db /= 12.92;

    X = dr * 0.4124 + dg * 0.3576 + db * 0.1805;
    Y = dr * 0.2126 + dg * 0.7152 + db * 0.0722;
    Z = dr * 0.0193 + dg * 0.1192 + db * 0.9505;
};
template<typename T_IN, typename T_OUT>
void xyz2rgb(T_IN const &X, T_IN const &Y, T_IN const &Z,
             T_OUT &R, T_OUT &G, T_OUT &B)
{
    float dr = X *  3.2406 + Y *(-1.5372)+ Z *(-0.4986);
    float dg = X *(-0.9689)+ Y *  1.8758 + Z *  0.0415;
    float db = X *  0.0557 + Y *(-0.2040)+ Z *  1.0570;

    if (dr > 0.0031308)
        dr = 1.055 * std::pow(dr, 1/2.4) - 0.055;
    else
        dr *= 12.92;
    if (dg > 0.0031308)
        dg = 1.055 * std::pow(dg, 1/2.4) - 0.055;
    else
        dg *= 12.92;
    if (db > 0.0031308)
        db = 1.055 * std::pow(db, 1/2.4) - 0.055;
    else
        db *= 12.92;

    R = dr * 255;
    G = dg * 255;
    B = db * 255;
};
template<typename T_IN, typename T_OUT>
void xyz2lab(T_IN const &X, T_IN const &Y, T_IN const &Z,
             T_OUT &L, T_OUT &a, T_OUT &b)
{
    auto dx = static_cast<float>(X);
    auto dy = static_cast<float>(Y);
    auto dz = static_cast<float>(Z);
    if (dx > 0.008856)
        dx = std::pow(dx, 1.f/3);
    else
        dx = 7.787 * dx + 16.f/116;
    if (dy > 0.008856)
        dy = std::pow(dy, 1.f/3);
    else
        dy = 7.787 * dy + 16.f/116;
    if (dz > 0.008856)
        dz = std::pow(dz, 1.f/3);
    else
        dz = 7.787 * dz + 16.f/116;
    L = 116 * dy - 16;
    a = 500 *(dx - dy);
    b = 200 *(dy - dz);
};
template<typename T_IN, typename T_OUT>
void lab2xyz(T_IN const &L, T_IN const &a, T_IN const &b,
             T_OUT &X, T_OUT &Y, T_OUT &Z)
{
    auto dy = static_cast<float>(L+16) / 116;
    auto dx = static_cast<float>(a) / 500 + dy;
    auto dz = dy - static_cast<float>(b) / 200;

    if (dy > 0.206893f)
        dy = std::pow(dy, 3);
    else
        dy = (dy - 16.f/116) / 7.787f;
    if (dx > 0.206893f)
        dx = std::pow(dx, 3);
    else
        dx = (dx - 16.f/116) / 7.787f;
    if (dz > 0.206893f)
        dz = std::pow(dz, 3);
    else
        dz = (dz - 16.f/116) / 7.787f;

    X = dx;
    Y = dy;
    Z = dz;
};
template<typename T_IN, typename T_OUT>
void rgb2lab(T_IN const &R, T_IN const &G, T_IN const &B,
             T_OUT &L, T_OUT &a, T_OUT &b)
{
    rgb2xyz(R, G, B, L, a, b);
    xyz2lab(L, a, b, L, a, b);
};
template<typename T_IN, typename T_OUT>
void lab2rgb(T_IN const &L, T_IN const &a, T_IN const &b,
             T_OUT &R, T_OUT &G, T_OUT &B)
{
    lab2xyz(L, a, b, R, G, B);
    xyz2rgb(R, G, B, R, G, B);
};
} // pcv::conversion

namespace kernel
{
double * median(std::size_t const &size)
{
    auto k_size = size * size;
    auto kernel = new double[k_size];
    std::fill_n(kernel, k_size, 1.0/k_size);
    return kernel;
} // pcv::kernel::median

double * gaussian(std::size_t const &size,
                  double sigma = 0)
{
    auto k_size = size * size;
    auto kernel = new double[k_size];

    // @see http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel
    if (sigma == 0)
        sigma = 0.3*((size-1)*0.5-1) + 0.8;

    auto den = -2*sigma*sigma;
    auto pad = size/2;
    auto sum = 0.0;
    for (decltype(k_size) i = 0; i < k_size; ++i)
    {
        auto r = i/size;
        auto c = i%size;
        kernel[i] = std::exp(((r-pad)*(r-pad)+(c-pad)*(c-pad))/den);
        sum += kernel[i];
    }
    std::for_each(kernel, kernel+k_size,
                  [&sum](double &k)
                  {
                      k /= sum;
                  }
                 );
    return kernel;
} // pcv::kernel::gaussian

} // pcv::kernel

namespace filter
{

template<typename T>
double * canny(const T * const &im, // must be grayscale
               int const &height,
               int const &width,
               int const &kernel_size = 5,
               double const &sigma = 0
              )
{
    auto im_size = height*width;

    static double PrewittX[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    static double PrewittY[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

    // Gaussian blurring
    auto kernel = pcv::kernel::gaussian(5);
    auto blurred_im = pcv::math::conv2(im, height, width, 1, 1, kernel, 5, 5);

    // Gradient
    auto grad_mag = new double[im_size];
    auto grad_dir = new double[im_size];
    auto h_grad = pcv::math::conv2(blurred_im, height, width, 1, 1, PrewittX, 3, 3);
    auto v_grad = pcv::math::conv2(blurred_im, height, width, 1, 1, PrewittY, 3, 3);
    for (auto i = 0; i < im_size; ++i)
    {
        grad_mag[i] = std::sqrt(h_grad[i] * h_grad[i] + v_grad[i]*v_grad[i]);
        grad_dir[i] = std::atan(v_grad[i]/h_grad[i]);
    }
    delete [] kernel;
    delete [] blurred_im;

    // Non-Maxima Suppression
    const double PI_by_4 = std::atan(1);
    const double PI_by_2 = std::asin(1);
    double theta1;
    for (auto h = 0, i = 0; h < height; ++h)
    {
        for (auto w = 0; w < width; ++w, ++i)
        {
            int p1 = -1, p2 = -1, p3 = -1, p4 = -1;
            if (grad_dir[i] >= -PI_by_2 && grad_dir[i] <= -PI_by_4)
            {
                if (h != height-1)
                {
                    if (w != width - 1)
                        p1 = i + width + 1;
                    p2 = i + width;
                }
                if (h != 0)
                {
                    if (w != 0)
                        p3 = i-width-1;
                    p4 = i-width;
                }
                theta1 = grad_dir[i] + PI_by_2;
            }
            else if (grad_dir[i] > -PI_by_4 && grad_dir[i] <= 0)
            {
                if (w != width-1)
                {
                    p1 = i+1;
                    if (h != height-1)
                        p2 = i+width+1;
                }
                if (w != 0)
                {
                    p3 = i-1;
                    if (h != 0)
                        p4 = i-width-1;
                }
                theta1 = grad_dir[i] + PI_by_4;
            }
            else if (grad_dir[i] > 0 && grad_dir[i] <= PI_by_4)
            {
                if (w != width-1)
                {
                    if (h != 0)
                        p1 = i-width+1;
                    p2 = i+1;
                }
                if (w != 0)
                {
                    if (h != height-1)
                        p3 = i+width-1;
                    p4 = i-1;
                }
                theta1 = grad_dir[i];
            }
            else //if (grad_dir[i] > PI_by_4 && grad_dir[i] <= PI_by_2)
            {
                if (h != 0)
                {
                    p1 = i-width;
                    if (w != width-1)
                        p2 = i-width+1;
                }
                if (h != height-1)
                {
                    p3 = i+width;
                    if (w != 0)
                        p4 = i+width-1;
                }
                theta1 = grad_dir[i] - PI_by_4;
            }
            theta1 = 1/std::tan(theta1);
            double max1 = 0, max2 = 0;
            if (p2 != -1)
                max1 = theta1 * grad_mag[p2];
            if (p1 != -1)
                max1 += (1-theta1)*grad_mag[p1];
            if (p4 != -1)
                max2 = theta1 * grad_mag[p4];
            if (p3 != -1)
                max2 += (1-theta1)*grad_mag[p3];
            if (grad_mag[i] < std::max(max1, max2))
                grad_mag[i] = 0;
        }
    }
    delete [] grad_dir;

    // Hysteresis
    auto edge = new double[im_size];
    double min = 1000000, max = 0;
    const double one_by_256 = 0.00390625;
    for (auto i = 0; i < im_size; ++i)
    {
        if (grad_mag[i] < min and grad_mag[i] > one_by_256)
            min = grad_mag[i];
        if (grad_mag[i] > max)
            max = grad_mag[i];
    }
    auto gap = max - min;
    auto upper_threshold = min + gap * 0.5;
    auto lower_threshold = min + gap * 0.1;
    for (auto i = 0; i < im_size; ++i)
    {
        auto flag = false;
        if (grad_mag[i] >= lower_threshold)
            flag = true;
        else
        {
            auto r = i/width;
            auto c = i%width;
            for (auto i = std::max(0, r-1); i < std::min(r+1, height); ++i)
            {
                for (auto j = std::max(0, c-1); j < std::min(c+1, width); ++j)
                {
                    if (grad_mag[i*width+j] > upper_threshold)
                    {
                        flag = true;
                        break;
                    }
                }
                if (flag)
                    break;
            }
        }
        if (flag)
            edge[i] = grad_mag[i];
        else
            edge[i] = 0;
    }
    delete [] grad_mag;
    return edge;


}; // pcv::filter::canny

template<typename T_IN, typename T_OUT>
void meanShift(const T_IN * const &src,
               std::size_t const &height,
               std::size_t const &width,
               std::size_t const &channel,
               std::size_t const &stride,
               double sp,
               double sr,
               T_OUT * const &dst)
{
    if (sp < 1) sp = 1.;
    if (sr < 1) sr = 1.;

    auto max_iter = 5;
    auto eps = 1;

    auto rsp = static_cast<std::size_t>(std::round(sp));
    auto ssp = static_cast<std::size_t>(std::round(sp*sp));
    auto ssr = static_cast<int>(std::round(sr*sr));

    T_OUT *dptr = src == reinterpret_cast<T_IN*>(dst) ? (new T_OUT[height*width*stride]) : dst;

    int dist_map[256];
    for (auto i = 0; i < 256; ++i)
        dist_map[i] = i*i;

#pragma omp parallel for num_threads(8)
    for (std::size_t r = 0; r < height; ++r)
    {
        std::vector<double> sum(channel, 0);
        std::vector<double> color(channel, 0);

        for (std::size_t c = 0; c < width; ++c)
        {
            std::size_t y0 = r, y;
            std::size_t x0 = c, x;
            int count;

            auto s = stride * (r * width + c);
            for (std::size_t ch = 0; ch < channel; ++ch)
                color[ch] = src[s + ch];

            for (auto i = 0; i < max_iter; ++i)
            {
                y = y0;
                x = x0;
                count = 0;

                auto sum_h = 0.f;
                auto sum_w = 0.f;

                std::fill_n(sum.begin(), channel, 0);

                for (std::size_t h = y > rsp ? y - rsp : 0,
                             h_end = std::min(y + rsp + 1, height);
                     h < h_end; ++h)
                {
                    for (std::size_t w = x > rsp ? x - rsp : 0,
                                 w_end = std::min(x + rsp + 1, width);
                         w < w_end; ++w)
                    {
                        if ((h - y) * (h - y) + (w - x) * (w - x) > ssp)
                            continue;

                        auto t = stride * (h * width + w);
                        auto dist = 0;
                        for (std::size_t ch = 0; ch < channel; ++ch)
                            dist += dist_map[static_cast<int>(src[t + ch] - color[ch])];

                        if (dist > ssr)
                            continue;

                        for (std::size_t ch = 0; ch < channel; ++ch)
                            sum[ch] += src[t + ch];

                        sum_h += h;
                        sum_w += w;
                        ++count;
                    }
                }

                if (count == 0)
                    break;

                y0 = std::round(sum_h / count);
                x0 = std::round(sum_w / count);

                for (std::size_t ch = 0; ch < channel; ++ch)
                    color[ch] = sum.at(ch) / count;

                if (std::abs(y - y0) < eps && std::abs(x - x0) < eps)
                    break;
            }

            s = channel * (r * width + c);
            for (std::size_t ch = 0; ch < channel; ++ch)
                dptr[s + ch] = static_cast<T_OUT>(color[ch]);
        }
    }


    if (src == reinterpret_cast<T_IN*>(dst))
        std::memcpy(dst, dptr, sizeof(T_OUT)*height*width*channel);

} // pcv::filter::meanshift

template<typename T_IN, typename T_OUT>
void watercolor(T_IN *org_im, // must be rgb
                std::size_t const &height,
                std::size_t const &width,
                std::size_t const &stride,
                T_OUT *new_im)
{
    constexpr static float MEAN[]   = {57.49f, -8.12f, 12.45f}; // MEAN and STDDEV can be obtained by learning actual paintings
    constexpr static float STDDEV[] = {23.45f, 7.26f, 20.85f};
    constexpr static unsigned char texture[] = {
#include "watercolor_canvas_texture.dat"
    };

    auto lab = new float[height*width*3];
    std::array<float, 3> mean{0,0,0}, stddev{0,0,0};
    auto tot = static_cast<int>(height*width);
    for (auto i = 0, s = 0, t = 0; i < tot; ++i, s+=stride, t+=3)
    {
        conversion::rgb2lab(org_im[s], org_im[s+1], org_im[s+2],
                            lab[t], lab[t+1], lab[t+2]);
        mean[0] += lab[t];
        mean[1] += lab[t+1];
        mean[2] += lab[t+2];
        stddev[0] += lab[t]*lab[t];
        stddev[1] += lab[t+1]*lab[t+1];
        stddev[2] += lab[t+2]*lab[t+2];
    }
    mean[0] /= tot;
    mean[1] /= tot;
    mean[2] /= tot;
    stddev[0] = STDDEV[0] / std::sqrt(stddev[0] / tot - mean[0]*mean[0]);
    stddev[1] = STDDEV[1] / std::sqrt(stddev[1] / tot - mean[1]*mean[1]);
    stddev[2] = STDDEV[2] / std::sqrt(stddev[2] / tot - mean[2]*mean[2]);
    for (auto i = 0, t = 0; i < tot; ++i, t+=3)
    {
        lab[t]   = (lab[t]   - mean[0]) * stddev[0] + MEAN[0];
        lab[t+1] = (lab[t+1] - mean[1]) * stddev[1] + MEAN[1];
        lab[t+2] = (lab[t+2] - mean[2]) * stddev[2] + MEAN[2];
    }
    for (auto i = 0, t = 0; i < tot; ++i, t+=3)
    {
        conversion::lab2rgb(lab[t], lab[t+1], lab[t+2],
                            lab[t], lab[t+1], lab[t+2]);
    }

    // abstraction
    auto meanshift = new float[tot*3];
    auto abstract = new float[tot*3];
    meanShift(lab, height, width, 3, 3, 20, 20, meanshift);
    std::vector<double> sum(3, 0);
    std::size_t abstract_size = 5;
#pragma omp parallel for num_threads(8)
    for (std::size_t r=0; r < height; ++r)
    {
        for (std::size_t c = 0; c < width; ++c)
        {
            std::fill_n(sum.begin(), 3, 0);
            auto count = 0;
            auto s = (r*width + c)*3;

            for (std::size_t h = r > abstract_size ? r-abstract_size : 0, h_end = std::min(r+abstract_size+1, height);
                 h < h_end; ++h)
            {
                for (std::size_t w = c > abstract_size ? c-abstract_size : 0, w_end = std::min(c+abstract_size+1, width);
                     w < w_end; ++w)
                {
                    auto t = 3*(h*width+w);
                    if (std::abs(meanshift[t] - meanshift[s]) < 6 &&
                            std::abs(meanshift[t+1] - meanshift[s+1]) < 6 &&
                            std::abs(meanshift[t+2] - meanshift[s+2]) < 6)
                    {
                        sum[0] += lab[t];
                        sum[1] += lab[t+1];
                        sum[2] += lab[t+2];
                        ++count;
                    }
                }
            }

            abstract[s]   = static_cast<T_OUT>(sum[0]/count);
            abstract[s+1] = static_cast<T_OUT>(sum[1]/count);
            abstract[s+2] = static_cast<T_OUT>(sum[2]/count);
        }
    }
    delete [] meanshift;
    delete [] lab;

    // edge darken
    // @see http://blog.csdn.net/grafx/article/details/59108946
    auto icopy = new float[tot*3];
    std::memcpy(icopy, abstract, sizeof(float)*tot*3);
    auto strength = width/3.f;
    // @see https://en.wikipedia.org/wiki/Perlin_noise
    auto raw_noise = [] (int const &c, int const &r)
    {
        auto n = c + r *57;
        n = std::pow(n<<13, n);
        return 1.0f - ( (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f;
    };
    auto smooth_noise = [&](int const &c, int const &r)
    {
        return (raw_noise(c-1, r-1)+raw_noise(c+1, r-1)+raw_noise(c-1, r+1)+raw_noise(c+1, r+1))/16.f +
               (raw_noise(c-1, r)+raw_noise(c+1, r)+raw_noise(c, r-1)+raw_noise(c, r+1))/8.f +
                raw_noise(c, r)/4.f;
    };
    auto lerp = [](float const &a0, float const &a1, float const &w)
    {
        return (1 - w) * a0 + w * a1;
    };
    auto interpolate = [&](float const &x, float const &y)
    {
        auto x0 = static_cast<int>(std::floor(x));
        auto y0 = static_cast<int>(std::floor(y));
        auto dx = x - x0;
        auto dy = y - y0;
        return lerp(lerp(smooth_noise(x0, y0), smooth_noise(x0+1,y0), dx),
                    lerp(smooth_noise(x0, y0+1), smooth_noise(x0+1, y0+1), dx),
                    dy);
    };
    auto get_noise = [&](int const &x, int const &y)
    {
        auto tot = 0;
        for (auto i = 0; i < 10; ++i)
        {
            auto freq = 2*i;
            tot += interpolate(x*freq, y*freq) * pcv::math::PI;
        }
        return tot;
    };
    auto perlin = new float[tot];
#pragma omp parallel for num_threads(8)
    for (std::size_t r = 1; r < height; ++r)
    {
        for (std::size_t c = 1; c < width; ++c)
            perlin[r*width+c] = get_noise(c, r);
    }
    auto h_m_1 = static_cast<int>(height-1);
    auto w_m_1 = static_cast<int>(width-1);
#pragma omp parallel for num_threads(8)
    for (auto r = 1; r < h_m_1; ++r)
    {
        for (auto c = 1; c < w_m_1; ++c)
        {
            auto s = (r * width + c) * 3;

            auto x0 = c + (perlin[r*width+c+1] - perlin[r*width+c]) * strength;
            auto y0 = r + (perlin[(r+1)*width+c] - perlin[r*width+c]) * strength;

            auto x = static_cast<int>(x0);
            auto y = static_cast<int>(y0);
            auto dx = x0 - x;
            auto dy = y0 - y;

            auto v00 = (y*width + x)*3;
            auto v01 = x == w_m_1 ? v00 : v00 + 3;
            auto v10 = y == h_m_1 ? v00 : v00 + 3*width;
            auto v11 = y == h_m_1 ? v01 : v01 + 3*width;

            auto vh0 = icopy[v00] + (icopy[v01] - icopy[v00])*dx;
            auto vh1 = icopy[v10] + (icopy[v11] - icopy[v10])*dx;
            auto val = vh0 + (vh1 - vh0)*dy;
            abstract[s]   = icopy[s]*0.4+ val*0.6;

            ++v00, ++v01, ++v10, ++v11;
            vh0 = icopy[v00] + (icopy[v01] - icopy[v00])*dx;
            vh1 = icopy[v10] + (icopy[v11] - icopy[v10])*dx;
            val = vh0 + (vh1 - vh0)*dy;
            abstract[s+1] = icopy[s+1]*0.4+ val*0.6;

            ++v00, ++v01, ++v10, ++v11;
            vh0 = icopy[v00] + (icopy[v01] - icopy[v00])*dx;
            vh1 = icopy[v10] + (icopy[v11] - icopy[v10])*dx;
            val = vh0 + (vh1 - vh0)*dy;
            abstract[s+2] = icopy[s+2]*0.4+ val*0.6;
        }
    }
    delete [] perlin;

    // edge darken
    std::memcpy(icopy, abstract, sizeof(float)*tot*3);
    for (auto r = 1; r < h_m_1; ++r)
    {
        for (auto c = 1; c < w_m_1; ++c)
        {
            auto left = (r * width + c - 1) * 3;
            auto right = left + 6;
            auto up = ((r - 1) * width + c) * 3;
            auto down = up + 2 * width * 3;
            auto s = (r * width + c) * 3;
            auto grad = std::abs(icopy[left]-icopy[right]);
            grad += std::abs(icopy[left+1]-icopy[right+1]);
            grad += std::abs(icopy[left+2]-icopy[right+2]);
            grad += std::abs(icopy[up]-icopy[down]);
            grad += std::abs(icopy[up+1]-icopy[down+1]);
            grad += std::abs(icopy[up+2]-icopy[down+2]);
            grad /= 8.f;
            grad = std::min(grad, 0.299f*abstract[s]+0.587f*abstract[s+1]+0.114f*abstract[s+2]);

            abstract[s]   -= grad;
            abstract[s+1] -= grad;
            abstract[s+2] -= grad;
        }
    }
    delete [] icopy;

    // pigment dispersion
    std::random_device rd;
    std::mt19937 sd(rd());
    std::normal_distribution<double> dist(0, 2);
    std::for_each(abstract, abstract+height*width*3,
                  [&](float &p)
                  {
                      p += dist(sd);
                  });
    // texture
    auto s = 0;
    for (std::size_t r = 0; r < height; ++r)
    {
        for (std::size_t c = 0; c < width; ++c, s+=3)
        {
            auto t = (r%768)*1024 + (c%1024);
            new_im[s]   = abstract[s]*0.9f + 0.1f*texture[t];
            new_im[s+1] = abstract[s+1]*0.9f + 0.1f*texture[t];
            new_im[s+2] = abstract[s+2]*0.9f + 0.1f*texture[t];
        }
    }

    delete [] abstract;
} // pcv::filter::watercolor
} // pcv::filter

} // pcv

#endif // WATERCOLOR_HPP