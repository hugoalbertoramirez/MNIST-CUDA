#include <stdio.h>
#include <stdlib.h>
 

__host__ double** getWeightMatrix(int rows, int cols)
{
    FILE *myFile;
    myFile = fopen("WMatrix.txt", "r");

    double** mat;
    cudaMallocManaged(&mat, rows * sizeof(double*));

    for (int i = 0; i < rows; i++)
    {
        cudaMallocManaged(&(mat[i]), cols * sizeof(double));

        fscanf(myFile, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", 
            &(mat[i][0]), &(mat[i][1]), &(mat[i][2]), &(mat[i][3]), &(mat[i][4]),
            &(mat[i][5]), &(mat[i][6]), &(mat[i][7]), &(mat[i][8]), &(mat[i][9])
        );
    }
    fclose(myFile);

    return mat;
}

__host__ double* getBVector(int cols)
{
    FILE *myFile;
    myFile = fopen("bVector.txt", "r");

    double* vector;
    cudaMallocManaged(&vector, cols * sizeof(double));

    fscanf(myFile, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", 
        &(vector[0]), &(vector[1]), &(vector[2]), &(vector[3]), &(vector[4]),
        &(vector[5]), &(vector[6]), &(vector[7]), &(vector[8]), &(vector[9])
    );
    
    fclose(myFile);

    return vector;
}

__host__ void printMatrix(double **mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        printf("[");
        for (int j = 0; j < cols; j++)
        {
            printf("%0.10lf\t", mat[i][j]);
        }
        printf("]\n");
    }
    printf("\n");
}

__host__ void getMNISTTest(int indice, int* label, int* vector, int rows, int cols)
{
    if (indice > 10000)
    {
        printf("Solo hay 10,000 datos de test !");
        return;
    }

    FILE *myFile;
    myFile = fopen("mnist_test.csv", "r");

    for (int i = 0; i < indice; i++)
    {
        fscanf(myFile, "%*[^\n]\n");
    }

    fscanf(myFile, "%d", label);

    fscanf(myFile, ",");

    for (int j = 0; j < rows; j++)
    {
        fscanf(myFile, "%d,", &(vector[j]));
    }

    fclose(myFile);
}

__host__ void printImage(int label, int* vector, int rows)
{
    for (int j = 0, x; j < rows; j++)
    {
        x = vector[j];

        if (x == 0)
        {
            printf("--");
        }
        else
        {
            printf("%d%d", label, label);
        }

        if ((j + 1) % 28 == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
}

__host__ double getMaxIndex(double* y, int cols)
{
    double max = y[0];
    int maxIndex = -1;

    for (int i = 0; i < cols; i++)
    {
        if (y[i] >= max)
        {
            max = y[i];
            maxIndex = i;
        }
    }
    return maxIndex;   
}

__global__ void productMatrixVectorKernel(int* X, double** W, double* WX, int cols)
{
    int j = threadIdx.y;

    double sum = 0;
    for (int k = 0; k < cols; k++)
    {
        sum += X[k] * W[k][j];
    }
    WX[j] = sum;

    __syncthreads();
}

__global__ void SumVectorVectorKernel(double* WX, double* b, double* y)
{
    int j = threadIdx.y;
    
    y[j] = WX[j] + b[j];

    __syncthreads();
}

int main()
{
    int rows = 784;
    int cols = 10;
    int indexTest = 44;

    double** W = getWeightMatrix(rows, cols);
    double* b = getBVector(cols);

    int label;
    int* X; 
    cudaMallocManaged(&X, rows * sizeof(int));
    double* WX;
    cudaMallocManaged(&WX, cols * sizeof(double));
    double* y;
    cudaMallocManaged(&y, cols * sizeof(double));

    int prediction;

    while(1)
    {
        printf("Indice del Data Set: ");
        scanf("%d", &indexTest);
        if (indexTest > 10000)
        {
            printf("Solo hay 10,000 datos\n\n");
            continue;
        }

        getMNISTTest(indexTest, &label, X, rows, cols);

        dim3 blocksPerGrid(1);
        dim3 threadsPerBlock(1, cols);

        productMatrixVectorKernel<<< blocksPerGrid, threadsPerBlock >>>(X, W, WX, rows);
        cudaDeviceSynchronize();

        SumVectorVectorKernel<<< blocksPerGrid, threadsPerBlock >>>(WX, b, y);
        cudaDeviceSynchronize();

        prediction = getMaxIndex(y, cols);
        printf("Original %d, Prediccion %d\n", label, prediction);

        printImage(label, X, rows);
    }
}
