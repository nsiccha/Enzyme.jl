/* A private LP64 BLAS provider for the computed-library test. Each copy of the
   library counts calls to the primal, reverse cache, and adjoint separately. */
static int calls[3];

int blas_calls(int routine) { return calls[routine]; }

double ddot(const int *n, const double *x, const int *sx, const double *y,
            const int *sy) {
    double result = 0;
    ++calls[0];
    for (int i = 0; i < *n; ++i)
        result += x[i * *sx] * y[i * *sy];
    return result;
}

void dcopy(const int *n, const double *x, const int *sx, double *y,
           const int *sy) {
    ++calls[1];
    for (int i = 0; i < *n; ++i)
        y[i * *sy] = x[i * *sx];
}

void daxpy(const int *n, const double *a, const double *x, const int *sx,
           double *y, const int *sy) {
    ++calls[2];
    for (int i = 0; i < *n; ++i)
        y[i * *sy] += *a * x[i * *sx];
}
