
__global__ void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}

__host__ void hsaxpy(int n, float a, float *x, float *y)
{
	float* d_x;
	float* d_y;
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, n * sizeof(float));
	cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);
	saxpy<<<(n + 255) / 256, 256 >>>(n, 2.0, d_x, d_y);
	cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
}