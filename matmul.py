import numpy as np
import time

def read_csv_to_numpy(file_path):
    try:
        return np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        print(f"Error reading file {file_path} {e}")
        exit(1)


def main():
    a_mat_path = './a_mat.csv'
    b_mat_path = './b_mat.csv'
    c_mat_path = './c_mat.csv'

    mat_a = read_csv_to_numpy(a_mat_path);
    mat_b = read_csv_to_numpy(b_mat_path);
    mat_c = read_csv_to_numpy(c_mat_path);

    print(mat_a.shape, mat_b.shape, mat_c.shape)

    start_time = time.time()
    result = np.dot(mat_a, mat_b)
    end_time = time.time()
    
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken}")

    tolerance = 1e-3
    diff = np.abs(result - mat_c)
    print(f'Diff: {diff}')
    are_similar = np.all(diff <= tolerance)

    if are_similar:
        print("The arrays are similar within the given tolerance.")
    else:
        print("The arrays are not similar.")

if __name__ == "__main__":
    main()