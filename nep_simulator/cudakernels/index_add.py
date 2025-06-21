import cupy as cp
import numpy as np


def index_add_cupy(vals: cp.ndarray, inds: cp.ndarray, final_sum: cp.ndarray):
    """
    Performs an index_add operation using a custom CuPy CUDA kernel.

    This function simulates PyTorch's index_add operation where elements
    from `vals` are added to `final_sum` at indices specified by `inds`.
    It handles potential race conditions using atomic operations in the CUDA kernel.

    Args:
        vals (cp.ndarray): A 1D CuPy array of float values (N_large elements).
                           These are the values to be summed.
        inds (cp.ndarray): A 1D CuPy array of integer indices (N_large elements),
                           where each index is between 0 and N_small-1.
                           These specify the target locations in `final_sum`.
        final_sum (cp.ndarray): A 1D CuPy array of float values (N_small elements),
                                typically initialized to zeros. This array will
                                accumulate the sums.
    """
    N_large = vals.size
    N_small = final_sum.size

    # Define the CUDA kernel as a string.
    # The kernel takes `vals` (source values), `inds` (target indices),
    # `final_sum` (destination array), `N_large` (size of vals/inds),
    # and `N_small` (size of final_sum) as arguments.
    # 'extern "C" __global__' specifies a global CUDA kernel.
    index_add_kernel_code = """
    extern "C" __global__ void index_add_kernel(
        const float* vals,   // Input values array
        const int* inds,     // Input indices array
        float* final_sum,    // Output sum array (modified in-place)
        int N_large,         // Size of vals and inds arrays
        int N_small)         // Size of final_sum array
    {
        // Calculate a unique global index for the current thread.
        // blockIdx.x: x-dimension index of the current block within the grid.
        // blockDim.x: x-dimension size of each block (number of threads per block).
        // threadIdx.x: x-dimension index of the current thread within its block.
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if the current thread's index is within the bounds of the input arrays.
        // This is crucial for handling cases where N_large is not perfectly divisible
        // by the total number of threads launched.
        if (idx < N_large) {
            // Get the target index from the `inds` array for the current value.
            int target_idx = inds[idx];

            // Get the value to be added from the `vals` array.
            float value_to_add = vals[idx];

            // Perform an atomic addition to `final_sum` at the `target_idx`.
            // `atomicAdd` is essential here. It ensures that if multiple threads
            // attempt to write to the same memory location (`final_sum[target_idx]`)
            // concurrently, their operations are serialized, preventing race conditions
            // and ensuring correct accumulation of sums.
            // Without atomicAdd, the final result could be incorrect due to lost writes.
            atomicAdd(&final_sum[target_idx], value_to_add);
        }
    }
    """

    # Compile the CUDA kernel using cupy.RawKernel.
    # The first argument is the CUDA source code.
    # The second argument is the name of the kernel function defined in the source.
    cuda_kernel = cp.RawKernel(index_add_kernel_code, 'index_add_kernel')

    # Determine optimal grid and block dimensions for kernel launch.
    # A typical block size (threads per block) is 256 or 512 for modern GPUs.
    threads_per_block = 256
    # Calculate the number of blocks needed to cover all N_large elements.
    # The ceiling division (N_large + threads_per_block - 1) // threads_per_block
    # ensures that enough blocks are launched to process all elements.
    blocks_per_grid = (N_large + threads_per_block - 1) // threads_per_block

    # Launch the CUDA kernel.
    # The first tuple (blocks_per_grid,) specifies the grid dimensions.
    # The second tuple (threads_per_block,) specifies the block dimensions.
    # The remaining arguments are the actual CuPy arrays and integers
    # that are passed to the kernel function.
    cuda_kernel((blocks_per_grid,), (threads_per_block,), (vals, inds, final_sum, N_large, N_small))

    # Synchronize the CUDA stream.
    # This ensures that all operations launched on the default stream
    # (including our kernel) have completed before the Python function returns.
    # This is important if you want to immediately use the `final_sum` array
    # on the host CPU or in subsequent device operations.
    cp.cuda.Stream.null.synchronize()


# --- Example Usage ---
if __name__ == '__main__':
    # Define parameters
    N_large = 5000000  # A large number of values
    N_small = 1000     # A smaller number of accumulation bins

    print(f"Running index_add with N_large={N_large}, N_small={N_small}")

    # 1. Create sample CuPy arrays
    # Values to be summed (random floats between 0 and 1)
    vals_cpu = np.random.rand(N_large)
    # vals_cpu = cp.random.rand(N_large, dtype=cp.float32)
    
    # Indices where values should be summed (random integers between 0 and N_small-1)
    # inds_cpu = cp.random.randint(0, N_small, N_large, dtype=cp.int32)
    inds_cpu = np.random.randint(0, N_small, N_large, dtype=np.int32)
    
    # The array to accumulate sums, initialized to zeros
    final_sum_cpu = cp.zeros(N_small, dtype=cp.float32)

    # Copy arrays to GPU (device memory)
    vals_gpu = cp.asarray(vals_cpu, dtype=cp.float32)
    inds_gpu = cp.asarray(inds_cpu)
    final_sum_gpu = cp.asarray(final_sum_cpu)

    print("Executing custom CUDA kernel...")
    # Call the custom index_add function
    index_add_cupy(vals_gpu, inds_gpu, final_sum_gpu)
    print("Kernel execution complete.")

    # 2. Verify the result using a NumPy equivalent (for correctness check)
    # Note: np.add.at is a good way to simulate this on CPU,
    # but it's not parallelized in the same way for large datasets.
    print("Verifying result with NumPy equivalent (on CPU)...")
    expected_final_sum_cpu = cp.zeros(N_small, dtype=cp.float32)
    # Convert CuPy arrays back to NumPy for verification
    np_vals = vals_gpu.get()
    np_inds = inds_gpu.get()
    np_final_sum_expected = expected_final_sum_cpu.get()

    # Use np.add.at to perform the accumulation on CPU.
    # This is a safe way to do it on CPU, but less performant for large arrays.
    np.add.at(np_final_sum_expected, np_inds, np_vals)

    # Compare the GPU result with the CPU-calculated expected result
    tolerance = 1e-5 # Define a tolerance for floating-point comparisons
    # Transfer GPU result back to CPU for comparison
    actual_final_sum_cpu = final_sum_gpu.get()

    # Check if the results are close (due to floating point arithmetic)
    are_close = cp.allclose(actual_final_sum_cpu, np_final_sum_expected, rtol=tolerance, atol=tolerance)

    print(f"Results are close to NumPy equivalent: {are_close}")

    if not are_close:
        print("Discrepancy detected. Showing first 10 differing values:")
        diff_indices = cp.where(~cp.isclose(actual_final_sum_cpu, np_final_sum_expected, rtol=tolerance, atol=tolerance))[0]
        for i in diff_indices[:10]:
            print(f"Index {i}: GPU={actual_final_sum_cpu[i]:.6f}, CPU_Expected={np_final_sum_expected[i]:.6f}")
    else:
        print("Verification successful!")

    # You can print a few values to manually inspect
    # print("\nFirst 10 elements of final_sum_gpu:")
    # print(final_sum_gpu[:10])
    # print("\nFirst 10 elements of expected_final_sum_cpu:")
    # print(np_final_sum_expected[:10])

