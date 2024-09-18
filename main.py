import numpy as np
import time
import os
#np.show_config()
total_time = 0
itr = 5

# Set the number of OpenBLAS threads
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())

# Verify OpenBLAS thread count
print(f"OpenBLAS is using {os.cpu_count()} threads.")

for i in range(0, itr):
    # Initialize two 4000x4000 matrices with random values between -100 and 100
    A = np.random.uniform(-100, 100, (12000,12000)).astype(np.float32)
    B = np.random.uniform(-100, 100, (12000,12000)).astype(np.float32)

    # Record the start time
    start_time = time.time()

    # Perform the dot product (matrix multiplication)
    C = np.dot(A, B)

    # Record the end time and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    total_time += duration
    # Print the time taken for the dot product
    print(i, f"Time taken for dot product: {duration:.6f} seconds")

print('Total Time: ', total_time/itr)
