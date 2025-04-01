import os

omp_num_threads = os.environ.get('OMP_NUM_THREADS')
print("OMP_NUM_THREADS:", omp_num_threads)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

