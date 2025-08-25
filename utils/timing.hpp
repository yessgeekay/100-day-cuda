#ifndef TIMING_HPP
#define TIMING_HPP

#include <chrono>
#include <utility> // For std::forward
#include <cuda_runtime.h>

/**
 * @brief Times a standard C++ function on the host (CPU).
 *
 * @tparam Func The type of the callable function/functor.
 * @tparam Args The types of the arguments to the function.
 * @param func The function to execute and time.
 * @param args The arguments to pass to the function.
 * @return The elapsed time as a std::chrono::duration.
 */
template<typename Func, typename... Args>
auto time_it_host(Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    // Use std::forward to perfectly forward arguments
    std::forward<Func>(func)(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

/**
 * @brief Times a CUDA kernel launch on the device (GPU).
 *
 * This function correctly synchronizes the device to ensure the kernel
 * has finished before stopping the timer.
 *
 * @tparam Func The type of the callable function/functor (usually a lambda launching the kernel).
 * @tparam Args The types of the arguments to the function.
 * @param func The function that launches the CUDA kernel.
 * @param args The arguments to pass to the function.
 * @return The elapsed time as a std::chrono::duration.
 */
template<typename Func, typename... Args>
auto time_it_device(Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<Func>(func)(std::forward<Args>(args)...);
    
    // This is CRITICAL for accurate GPU timing.
    // It blocks the CPU thread until all previously issued GPU commands are complete.
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

#endif // TIMING_HPP
