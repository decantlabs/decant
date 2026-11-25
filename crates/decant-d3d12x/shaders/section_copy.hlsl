// Copies 32 bit words from a root SRV to a root UAV. d3d12xs_test uses it to read title memory
// from the GPU through the window of title_mem. Compiled at run time with D3DCompile (cs_5_1).
// `words` is the exact number of words to copy. The dispatch rounds up to whole groups of 64
// threads, so the threads past the last word must neither read nor write.
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

cbuffer Args : register(b0)
{
    uint words;
};

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= words)
    {
        return;
    }
    dst.Store(id.x * 4, src.Load(id.x * 4));
}