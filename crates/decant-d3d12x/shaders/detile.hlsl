// Reads the console tiled bytes of one mip level of a texture (all its array slices) from title
// memory and writes them row major into a staging buffer, which the caller then copies into the PC
// texture with CopyTextureRegion. Compiled at run time with D3DCompile (cs_5_1), see detile.rs.
//
// Parameter block (b0), one block of 512 bytes per dispatch, words:
//   0 width in elements      1 height in elements     2 bytes per element     3 output row pitch
//   4 linear (1) or tiled    5 in mip tail            6 tail x                7 tail y
//   8 block width            9 block height           10 blocks per row       11 log2 block bytes
//   12 level offset          13 slice size            14 linear row pitch     15 output slice stride
//   16 output base offset    17 slices                18 term count
//   20 + 4 * i: output bit, x mask, y mask, z mask of equation term i
// One thread writes one 32 bit output word (bytes per element below 4: several elements) or one element.
cbuffer Params : register(b0) { uint4 v[32]; }
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

uint W(uint i) { return v[i >> 2][i & 3]; }

uint element_offset(uint x, uint y, uint z)
{
    uint z_base = z * W(13);
    if (W(4) == 1)
        return z_base + W(12) + y * W(14) + x * W(2);
    uint ex = x, ey = y;
    if (W(5) != 0) { ex += W(6); ey += W(7); }
    uint n = W(18), e = 0;
    for (uint i = 0; i < n; i++)
    {
        uint b = 20 + i * 4;
        uint parity = countbits(ex & W(b + 1)) + countbits(ey & W(b + 2)) + countbits(z & W(b + 3));
        e |= (parity & 1) << W(b);
    }
    uint bits = W(11);
    uint reduced = e & ((1u << bits) - 1);
    if (W(5) != 0)
        return z_base + reduced;
    uint block_index = (y / W(9)) * W(10) + x / W(8);
    return z_base + W(12) + (block_index << bits) + reduced;
}

[numthreads(16, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width = W(0), height = W(1), bpe = W(2);
    uint y = id.y, z = id.z;
    if (y >= height || z >= W(17)) return;
    uint row = W(16) + z * W(15) + y * W(3);
    if (bpe < 4)
    {
        uint per = 4 / bpe;
        uint x0 = id.x * per;
        if (x0 >= width) return;
        uint mask = (1u << (bpe * 8)) - 1;
        uint word = 0;
        for (uint k = 0; k < per; k++)
        {
            uint x = x0 + k;
            if (x < width)
            {
                uint o = element_offset(x, y, z);
                uint value = (src.Load(o & ~3u) >> ((o & 3) * 8)) & mask;
                word |= value << (k * bpe * 8);
            }
        }
        dst.Store(row + id.x * 4, word);
    }
    else
    {
        uint x = id.x;
        if (x >= width) return;
        uint o = element_offset(x, y, z);
        for (uint i = 0; i < bpe / 4; i++)
            dst.Store(row + x * bpe + i * 4, src.Load(o + i * 4));
    }
}
