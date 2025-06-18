import cupy as cp
import numpy as np
import time

# CUDA kernel to sum 10 arrays along the last dimension
sum_10_arrays_kernel = cp.RawKernel(r'''
extern "C" __global__
void sum_10_arrays(const float* const* inputs, float* output, int Np, int dim1, int dim2) {
    /*
     * This kernel sums 10 arrays along the last dimension (dim2).
     *
     * inputs: An array of 10 pointers, each pointing to the start of a 3D input array on the GPU.
     * output: A pointer to the 3D output array on the GPU.
     * Np: The size of the first dimension.
     * dim1: The size of the second dimension.
     * dim2: The size of the third dimension (the one we are summing over).
     *
     * The grid is structured as a 2D grid of thread blocks, corresponding to the Np and dim1 dimensions.
     * Each thread block is 1D and will handle the summation for one "row" in the (Np, dim1) plane.
     */

    // Calculate the global thread indices for the first two dimensions
    int np_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d1_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to ensure we don't go out of bounds for Np and dim1
    if (np_idx >= Np || d1_idx >= dim1) {
        return;
    }

    // This kernel is designed to work with 10 input arrays.
    const int num_arrays = 10;

    // Loop through each of the 10 input arrays
    for (int i = 0; i < num_arrays; ++i) {
        // Pointer to the current input array being processed
        const float* current_input = inputs[i];
        float sum = 0.0f;

        // Each thread calculates the sum for a single row (np_idx, d1_idx) over the dim2 dimension.
        for (int d2_idx = 0; d2_idx < dim2; ++d2_idx) {
            // Calculate the linear index for the element in the 3D input array
            int linear_idx = np_idx * dim1 * dim2 + d1_idx * dim2 + d2_idx;
            sum += current_input[linear_idx];
        }

        // Calculate the linear index for the element in the 2D output array
        // The output shape is (10, Np, dim1)
        int output_idx = i * Np * dim1 + np_idx * dim1 + d1_idx;
        output[output_idx] = sum;
    }
}
''', 'sum_10_arrays')

# --- Parameters ---
Np = 6000
dim1 = 36
dim2 = 78
num_arrays = 10
Nloop = 2000
# --- Create Mock Data on GPU ---
print("Creating mock data on the GPU...")
# Create a list of 10 CuPy arrays with the specified shape
# input_arrays = [cp.random.rand(Np, dim1, dim2, dtype=cp.float32) for _ in range(num_arrays)]
input_arrays = []
for i in range(10):
    v = np.random.rand(Np,dim1,dim2)
    v = cp.asarray(v,dtype=cp.float32)
    input_arrays.append(v)


# --- Method 1: Using cp.sum in a loop (the original approach) ---
print("\nRunning benchmark for cp.sum...")
output_cupy = [cp.empty((Np, dim1), dtype=cp.float32) for _ in range(num_arrays)]


for k in range(5):
    for i in range(num_arrays):
        output_cupy[i] = cp.sum(input_arrays[i], axis=-1)
cp.cuda.Stream.null.synchronize()
start_time_cupy = time.time()
for k in range(Nloop):
    for i in range(num_arrays):
        output_cupy[i] = cp.sum(input_arrays[i], axis=-1)
# Synchronize the device to ensure all operations are complete before stopping the timer
cp.cuda.Stream.null.synchronize()
end_time_cupy = time.time()
duration_cupy = end_time_cupy - start_time_cupy
print(f"Time taken with cp.sum: {duration_cupy:.6f} seconds")


# --- Method 2: Using the custom CUDA kernel ---
print("\nRunning benchmark for the custom CUDA kernel...")

# Prepare inputs and output for the kernel
# The kernel expects an array of pointers to the input arrays.
input_pointers = cp.array([arr.data.ptr for arr in input_arrays], dtype=cp.uint64)
# The output will be a single large array of shape (10, Np, dim1)
output_kernel = cp.empty((num_arrays, Np, dim1), dtype=cp.float32)

# --- Kernel Launch Configuration ---
# Define the number of threads per block
threads_per_block = (16, 16)
# Calculate the grid dimensions needed to cover the Np and dim1 dimensions
grid_dim_x = (Np + threads_per_block[0] - 1) // threads_per_block[0]
grid_dim_y = (dim1 + threads_per_block[1] - 1) // threads_per_block[1]
grid = (grid_dim_x, grid_dim_y)


for k in range(5):
# Launch the kernel
    sum_10_arrays_kernel(
        grid,
        threads_per_block,
        (input_pointers, output_kernel, Np, dim1, dim2)
    )
cp.cuda.Stream.null.synchronize()
start_time_kernel = time.time()
for k in range(Nloop):
# Launch the kernel
    sum_10_arrays_kernel(
        grid,
        threads_per_block,
        (input_pointers, output_kernel, Np, dim1, dim2)
    )
# Synchronize the device
cp.cuda.Stream.null.synchronize()
end_time_kernel = time.time()
duration_kernel = end_time_kernel - start_time_kernel
print(f"Time taken with custom kernel: {duration_kernel:.6f} seconds")


# --- Verification ---
print("\nVerifying results...")
# Reshape the kernel's output to match the list structure of the cupy output
output_kernel_reshaped = [output_kernel[i] for i in range(num_arrays)]

all_close = True
for i in range(num_arrays):
    if not cp.allclose(output_cupy[i], output_kernel_reshaped[i], atol=1e-5):
        print(f"Mismatch found in array {i}")
        all_close = False
        break

if all_close:
    print("Verification successful: The results from both methods are identical.")
else:
    print("Verification failed: The results do not match.")

# --- Performance Comparison ---
print("\n--- Performance Summary ---")
print(f"Speedup from custom kernel: {duration_cupy / duration_kernel:.2f}x")
