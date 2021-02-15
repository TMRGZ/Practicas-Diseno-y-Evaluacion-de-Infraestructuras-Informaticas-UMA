// Includes
#include <stdio.h>

// Variables
float* h_A;   // Punteros para los vectores del host (CPU)
float* h_B;
float* h_C;
float* d_A;   // Punteros para los vectores del device (GPU)
float* d_B;
float* d_C;

// Funciones de apoyo (que hemos colocado al final de este archivo)
void Cleanup(void);
void RandomInit(float*, int);

// Codigo del DEVICE (kernel CUDA), que en otros ejercicios ubicaremos en un fichero aparte
__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Descomentar la siguiente línea si queremos monitorizar en pantalla la traza de ejecución por hilos
    // printf( "Thread ID: %d. Block ID: %d. Actualizando el indice: %d\n", threadIdx.x, blockIdx.x, i );

    if (i < N)
        C[i] = A[i] + B[i];
}

// Codigo del HOST
int main(int argc, char** argv)
{
    int N = 25600;  // Elegimos un tamaño de problema que ejecute exactamente 100 bloques de 256 hilos
    printf("Ejecutando una suma de vectores en CUDA con %d elementos\n", N);
    size_t size = N * sizeof(float);

    // Reservamos memoria para h_A y h_B en memoria del host
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup();
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup();
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup();
    
    // Inicializamos los vectores de entrada con valores aleatorios (ver función anexa al final de este fichero)
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Reservamos memoria para los vectores en el dispositivo
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copiamos los vectores desde el host al dispositivo
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invocamos el kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);   // Esta es la llamada al kernel

    // Copiamos el resultado desde el device (GPU) al host (CPU)
    // h_C contendra el resultado en memoria del host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Comprobamos el resultado comparando los valores que computa la GPU con los obtenidos en CPU
    int i;
    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];
        if (fabs(h_C[i] - sum) > 1e-5)
            break;
    }
    printf("%s \n", (i == N) ? "RESULTADOS CORRECTOS Y VALIDADOS CON LA CPU" : "RESULTADOS INCORRECTOS. NO COINCIDEN CON LOS OBTENIDOS POR LA CPU TRAS LA EJECUCIÓN SECUENCIAL DEL CÓDIGO");
    
    Cleanup();
}

void Cleanup(void)
{
    // Liberamos memoria del dispositivo
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Liberamos memoria del host
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);

    cudaThreadExit();
    
    exit(0);
}

// Inicializa un array con valores aleatorios flotantes.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

