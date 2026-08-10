// Shared prologue for every compute shader in this backend.
//
// Vulkan consumes SPIR-V, which is compiled from these GLSL sources by glslc.
// They are compiled ahead of time into src/evm/vulkan/shaders/*.spv and shipped
// in the package, so installing needs no shader compiler.
//
// Border handling is `reflect1`, matching the reference: index -1 reads
// element 1, and index n reads element n-2.

int reflect_index(int i, int n) {
    if (n == 1) return 0;
    int period = 2 * n - 2;
    int m = i % period;
    if (m < 0) m += period;
    return (m < n) ? m : (period - m);
}
