#include <algorithm>
#include <functional>
#include <omp.h>

void parallelFor(float* data, int size, std::function<void(float)> func)
{
    const int threads = omp_get_max_threads();
    const int step = (size + threads - 1) / threads;

#pragma omp parallel
    {
        const int t = omp_get_thread_num();
        const int begin = t * step;
        const int end = std::min((t + 1) * step, size);

        for (int i = begin; i < end; ++i)
        {
            func(data[i]);
        }
    }
}

void detrend(float* y, int size)
{
    // calculate mean values
    const float x_mean = (size - 1) / 2;
    float y_mean = 0;
    for (int i = 0; i < size; i++)
    {
        y_mean += y[i];
    }
    y_mean /= size;

    // calculate covariance and gradient
    float tmp1 = 0;
    float tmp2 = 0;
    for (int i = 0; i < size; i++)
    {
            tmp1 += y[i] * i;
            tmp2 += i * i;
    }
    const float sxy = tmp1 / size - x_mean * y_mean;
    const float sxx = tmp2 / size - x_mean * x_mean;
    const float grad = sxy / sxx;

    // calculate intercept
    const float y_int = -grad * x_mean + y_mean;

    // remove trend
    for (int i = 0; i < size; i++)
        y[i] -= grad * i + y_int;
}
