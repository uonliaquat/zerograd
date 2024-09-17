#include <blis/blis.h>
#include <stdio.h>

int main()
{
    // Initialize BLIS
    bli_init();

    // Declare scalars
    obj_t alpha, beta;

    // Declare matrices and vectors
    obj_t A, x, y;

    // Initialize scalars alpha and beta
    bli_obj_scalar_init_detached(BLIS_DOUBLE, &alpha);
    bli_obj_scalar_init_detached(BLIS_DOUBLE, &beta);

    // Set values of alpha and beta
    bli_setsc( 1.0, 0.0, &alpha ); // alpha = 1.0
    bli_setsc( 1.0, 0.0, &beta );  // beta = 1.0

    // Initialize matrix A and vectors x, y
    bli_obj_create( BLIS_DOUBLE, 3, 3, 0, 0, &A ); // 3x3 matrix
    bli_obj_create( BLIS_DOUBLE, 3, 1, 0, 0, &x ); // 3x1 vector
    bli_obj_create( BLIS_DOUBLE, 3, 1, 0, 0, &y ); // 3x1 vector

    // Fill matrix A and vector x with some values
    double A_data[9] = { 1.0, 2.0, 3.0,
                         4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0 };

    double x_data[3] = { 1.0, 1.0, 1.0 };
    double y_data[3] = { 0.0, 0.0, 0.0 };

    // Attach buffers to matrix A and vectors x, y (correct number of arguments)
    bli_obj_attach_buffer(A_data, 1, 3, 3, &A);  // Attach data to matrix A (matrix is 3x3)
    bli_obj_attach_buffer(x_data, 1, 3, 1, &x);  // Attach data to vector x (vector is 3x1)
    bli_obj_attach_buffer(y_data, 1, 3, 1, &y);  // Attach data to vector y (vector is 3x1)

    // Perform matrix-vector multiplication: y = alpha * A * x + beta * y
    bli_gemv(&alpha, &A, &x, &beta, &y);

    // Output result stored in y
    printf("Resulting vector y:\n");
    for (int i = 0; i < 3; i++) {
        printf("%f\n", y_data[i]);
    }

    // Finalize BLIS
    bli_finalize();

    return 0;
}