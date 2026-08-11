"""Pin BLAS to one thread BEFORE numpy is imported.

On this machine (numpy 1.26 + MKL) a tall-skinny ``(d, N) @ (N, d)`` product with N ~ 4e4
and d ~ 12 takes ~18 s with 10 BLAS threads and 5 ms with one -- a thread-oversubscription
pathology, not a property of the algorithms.  Every experiment therefore pins BLAS to a
single thread and parallelises over Monte Carlo replicates instead, which is also the
right setting for the wall-clock comparisons we report.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
