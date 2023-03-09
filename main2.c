#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>

#define VECTOR_SIZE 27000
#define TAU 0.00001

void vector_fill(double * vector, double fill_value) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = fill_value;
    }
}

void matrix_fill(double * matrix){
    for (int i = 0; i < VECTOR_SIZE; i++) {
        for (int j = 0; j < VECTOR_SIZE; j++) {
            if (i == j)
                matrix[i * VECTOR_SIZE + j] = 2;
            else
                matrix[i * VECTOR_SIZE + j] = 1;
        }
    }
}

double parallel_result_calculation(double* matrix, double* vectorX, double* vectorB, double* result) {
    int i, j; // Loop variables
    double norma = 0.0, dop = 0.0;
#pragma omp parallel for shared(matrix, vectorX, vectorB, result) private(i,j) reduction(+:norma) reduction(+:dop)
        for (i = 0; i < VECTOR_SIZE; i++) {
            result[i] = -1 * vectorB[i];
            for (j = 0; j < VECTOR_SIZE; j++)
                result[i] += matrix[i * VECTOR_SIZE + j] * vectorX[j];
            norma += result[i] * result[i];
            dop += vectorB[i] * vectorB[i];
            result[i] = vectorX[i] - TAU * result[i];
        }

    return sqrt(norma) / sqrt(dop);
}


int main() {
    double* matrix = (double *) malloc(VECTOR_SIZE * VECTOR_SIZE * sizeof(double));
    double* vectorX = (double *) malloc(VECTOR_SIZE * sizeof(double));
    double* vectorB = (double *) malloc(VECTOR_SIZE * sizeof(double));
    double* result = (double *) malloc(VECTOR_SIZE * sizeof(double));
    matrix_fill(matrix);
    vector_fill(vectorB, VECTOR_SIZE + 1);
    double min_time = 80.0;
    for (int k = 0; k < 10; k++) {
        vector_fill(vectorX, 0);

        double norma = 1;
        double e = 0.00001;
        double t1 = omp_get_wtime();

        while (norma >= e) {
            norma = parallel_result_calculation(matrix, vectorX, vectorB, result);
            for (int i = 0; i < VECTOR_SIZE; i++)
                vectorX[i] = result[i];
        }

        double t2 = omp_get_wtime();
        if (t2 - t1 < min_time)
            min_time = t2 - t1;
    }
    printf("time = %10.5f\n", min_time);

    for (int i = 0; i < 100 && i < VECTOR_SIZE; i++)
        printf("%10.5f\n", result[i]);

    free(vectorX);
    free(result);
    free(matrix);
    free(vectorB);
    return 0;
}
