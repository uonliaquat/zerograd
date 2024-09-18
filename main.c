#include <blis/blis.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>  // For sysconf()


float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

int main() {
    srand((unsigned int)time(NULL));

    // Initialize BLIS
    bli_init();

    obj_t a1, a2, c;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;

	dt = BLIS_FLOAT;
	m = 12000; n = 12000; rs = 1; cs = m;



      // Get the number of available processors/cores
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);

    // Set the number of threads to the maximum available cores
    bli_thread_set_num_threads(14);

    // Query and print the number of threads BLIS is using
    dim_t num_threads = bli_thread_get_num_threads();
    printf("BLIS is using %ld threads (max available: %ld).\n", num_threads, num_cores);

    int itr = 5;
    float total_time_taken = 0;
    for(int i = 0; i < itr; i++){
        // First we allocate and initialize a matrix by columns.
        double* p1 = malloc( m * n * sizeof( double ) );
        for(int i = 0; i < m * n; i++){
            p1[i] = float_rand(-100.0, 100.0);
        }

        bli_obj_create_without_buffer( dt, m, n, &a1 );
        bli_obj_attach_buffer( p1, rs, cs, 0, &a1 );
        //bli_printm( "matrix 'a1'", &a1, "%5.1f", "" );


        double* p2 = malloc( m * n * sizeof( double ) );
        for(int i = 0; i < m * n; i++){
            p2[i] = float_rand(-100.0, 100.0);
        }
        bli_obj_create_without_buffer( dt, m, n, &a2 );
        bli_obj_attach_buffer( p2, rs, cs, 0, &a2 );
        //bli_printm( "matrix 'a2'", &a2, "%5.1f", "" );



        double* p3 = malloc( m * n * sizeof( double ) );
        for(int i = 0; i < m * n; i++){
            p3[i] = 0;
        }
        bli_obj_create_without_buffer( dt, m, n, &c );
        bli_obj_attach_buffer( p3, rs, cs, 0, &c );

        // Define variables for start and end times
        struct timespec start, end;

        // Get the start time
        clock_gettime(CLOCK_MONOTONIC, &start);
        // Perform matrix multiplication: C = alpha * A * B + beta * C
        bli_gemm(&BLIS_ONE, &a1, &a2, &BLIS_ZERO, &c );
    // Get the end time
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Calculate the time difference in nanoseconds and convert to milliseconds
        long seconds = end.tv_sec - start.tv_sec;
        long nanoseconds = end.tv_nsec - start.tv_nsec;
        double milliseconds = seconds * 1000.0 + nanoseconds / 1.0e6;

        // Print the time taken in milliseconds
        printf("Time taken: %.3f milliseconds\n", milliseconds);
        total_time_taken += milliseconds;
    }
    printf("Total Time Take: %.3f milliseconds", total_time_taken/itr);
    // Print the result of the matrix multiplication
    //bli_printm( "Matrix C (result of A * B)", &c, "%4.1f", "" );

    // Finalize BLIS
    bli_finalize();



    return 0;
}
