// WEFT tile scoring kernel skeleton.
// Placeholder for CUDA raster scoring path referenced by encoder.py.

extern "C" __global__ void weft_score_tiles(
    const float* src_tiles,
    const float* primitive_params,
    float* out_mse,
    int tile_count,
    int tile_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_count) return;

    // TODO: Implement primitive raster scoring for candidate search.
    out_mse[idx] = 0.0f;
}
