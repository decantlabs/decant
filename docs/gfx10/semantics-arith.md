# GFX10 (RDNA2) arithmetic and logic semantics table

Companion to `semantics-arith.tsv` (600 rows, one per instruction, tab separated, 15 columns, header in line 1).
Target is gfx1030 (RDNA2). The table also covers the other GFX10 chips and flags the instructions that RDNA2 dropped or added.

## 1. Sources and clean room statement

Only these were read: Mesa (MIT) at `src/amd/compiler`, the LLVM AMDGPU backend (Apache 2 with LLVM exception) at `llvm/lib/Target/AMDGPU` plus its MC tests in `llvm/test/MC/AMDGPU`, and public AMD and Microsoft documentation from general knowledge.
No console emulator or translator project was opened.

The AMD RDNA2 ISA PDF was not available while this table was written. Every statement that comes from the public AMD ISA documents as recalled, instead of from the Mesa or LLVM text, is marked `K` in the `basis` column. Statements that are visible in the Mesa or LLVM files are marked `T` and carry a file and line in `source`. Treat `K` rows as a draft that still has to be checked against the PDF, and treat anything listed in section 9 as open.

Where the instruction list comes from: the GFX10 opcode column of `aco_opcodes.py` (instruction exists when the gfx10 number is not -1), cross checked against the `Real_gfx10` definitions in the LLVM `.td` files and against the GFX10 MC tests. Two ACO entries that LLVM does not accept for GFX10 were dropped (`s_mov_fed_b32`, `v_interp_p2_hi_f16`). ACO uses GFX6 style names for some GFX10 instructions, the `mnemonic` column uses the assembler (LLVM) name and `aco_name` keeps the ACO name. Examples: `v_add_u32` is `v_add_nc_u32`, `v_addc_co_u32` is `v_add_co_ci_u32`, `v_cmp_lg_i32` is `v_cmp_ne_i32`, `v_cmp_tru_u32` is `v_cmp_t_u32` (float forms keep `lg` and `tru`).

VOP3P (packed math such as `v_pk_add_f16`), VOPD, VINTRP (`v_interp_p1_f32` and so on), VOP3P dot products and memory formats are out of scope and have no rows. The three VOP3 form `v_interp_*_f16` instructions are listed because ACO tags them VOP3.

Rows per format: SOP1 65, SOP2 51, SOPK 27, SOPC 18, VOP1 81, VOP2 51, VOP3 117, VOPC 190.
Instructions with a VOP2, VOP1 or VOPC base encoding also exist in VOP3 (e64) encoding, that is one row, the `encodings` column lists the forms.

## 2. TSV columns

| column | meaning |
|---|---|
| mnemonic | GFX10 assembler name |
| format | base format class (SOP1, SOP2, SOPK, SOPC, VOP1, VOP2, VOP3, VOPC) |
| aco_name | name in `aco_opcodes.py` |
| availability | all GFX10, or only gfx1010 to gfx1013, or only gfx1030 and later |
| encodings | forms: vop1/vop2/vopc (32 bit), vop3, sdwa, dpp16, dpp8. Derived from the Mesa rules in `aco_ir.cpp:303-368` (SDWA) and `aco_ir.cpp:420-526` (DPP) |
| aco_operands | `sources > destinations`, types from ACO: u32 i32 f32 and so on, `@scc @exec @m0` fixed registers, `mask` = lane mask, trailing `m` on a source = accepts abs and neg |
| reads, writes | implicit state read and written (SCC, EXEC, M0, lane masks). All VALU instructions read EXEC as lane enable |
| semantics | the formal expression, see section 3 |
| edge_cases | the details that matter for exact results |
| hlsl | HLSL fragment, see section 5 |
| hlsl_notes | where HLSL defaults differ and what to write instead |
| basis | `T` seen in Mesa or LLVM text (see source), `K` from AMD ISA knowledge not verifiable in the clones |
| not_pinned_down | what could not be established |
| source | file and line in the clones |

## 3. Notation for the semantics column

* `S0 S1 S2` source operands in encoding order, `D` destination, `K` literal constant, `imm16` the SOPK immediate. `SDST` and `MASK` are lane masks.
* Typed views: `S0.u32`, `S0.i32`, `S0.f32`, `S0.f16`, `S0.u64`, `S0.f64`. A view never converts, it only reinterprets the bits. `S0.u32[4:0]` is a bit slice, `S0.u32[7]` one bit, `{a, b}` concatenation with `a` in the high bits.
* `sext(x, n)` sign extends from bit n-1, `zext33(x)` zero extends into a wider integer so that a carry is visible, `mask(w)` is `(1 << w), 1` with the width taken exactly as written.
* All integer arithmetic wraps modulo the operand width unless a saturation is stated. `(wrap)` repeats this where it matters.
* Float operations are IEEE 754 operations on the stated width, rounding per MODE (default round to nearest even), denormals per MODE (section 4).
* `##` separates statements that all happen as one instruction. Statements on one side do not depend on the others unless the right side names a value computed on the left (`t`, `cond`).
* Scalar instructions (SOP*) run once per wave. VALU instructions (VOP*) run once per lane `L` with `EXEC[L] == 1`, `S0` means the lane's value. Lanes with `EXEC == 0` keep their old destination value, except for lane mask destinations where the bit is written as 0 (see G3).
* `COUT`, `CIN`, `MASK` are lane masks: one bit per lane held in VCC, an SGPR pair (wave64) or one SGPR (wave32).
* `SCC` is the scalar condition bit, `EXEC` the lane enable mask, `VCC` the vector condition mask, `M0` the scalar misc register.
* Sources in the HLSL column: `a b c` are the operands S0 S1 S2 after source modifiers, `r` is the result D. `scc exec vcc cin cout mask` are booleans or masks. `k` is the literal, `imm` the SOPK immediate, `simm` the sign extended immediate. For unsigned scalar ops `a` and `b` are `uint` (or `uint64_t` for the 64 bit forms).

## 4. Global rules (apply to many rows)

**G1 Float modes.** MODE holds the rounding mode (separate for f32 and for f16/f64) and denormal flush bits (separate for f32 and for f16/f64). Mesa tracks them as `fp_mode.denorm32`, `denorm16_64` and the round fields (`aco_optimizer.cpp:3967`, `aco_select_nir_alu.cpp:3458`). A translator must follow `s_setreg_b32` / `s_denorm_mode` / `s_round_mode` writes. Default is round to nearest even. f32 denormal behaviour at program start is decided by the driver and cannot be assumed. When denormals are flushed, a denormal input counts as a zero of the same sign and a denormal result becomes a zero of the same sign. `v_mad_f32`, `v_mac_f32` flush always, the transcendental f32 instructions (`v_rcp_f32`, `v_log_f32`, `v_exp_f32` and so on) flush as described per row.

**G2 Source modifiers (float operands, VOP3, SDWA, DPP16).** `abs` clears the sign bit then `neg` flips it. They are applied on the bit pattern before the operation, also for `v_cndmask_b32` (this is how fabs and fneg are folded into a select). They are undefined or ignored for integer operations in the VOP3 form, do not apply them there.

**G3 Lane mask results.** Instructions that write a lane mask (`v_cmp_*`, carry outs of `v_add_co_*`, `v_div_scale_*`) write one bit per lane, lanes with EXEC off write 0. The VOPC e32 and VOP2 carry forms use VCC, the VOP3 forms use any SGPR pair (wave64) or one SGPR (wave32). `v_cmpx_*` writes only EXEC on GFX10 (`VOPCInstructions.td:2182-2200` has the `_nosdst_e64` form), it does not write VCC or an SGPR.

**G4 VOP3 output modifiers.** `clamp`: float results are clamped to [0, 1] and a NaN becomes 0 (Mesa relies on this: `aco_select_nir_alu.cpp:716-722`, "fract(Inf) is NaN, clamp turns it into zero"). Integer results saturate to the type range (`aco_select_nir_alu.cpp:1383-1407` builds `uadd_sat` from it). `omod` multiplies a float result by 2, 4 or 0.5 after the operation (f32 only mad style rules, denormal flush state matters, `aco_optimizer.cpp:4008`). Both are only available in the VOP3 form and in SDWA (GFX9 and later, `omod` GFX9 and later).

**G5 16 bit forms.** f16 and 16 bit integer VALU instructions write only 16 bits. On GFX10 the other half of the destination register is preserved for all of: the VOP2 and VOP1 f16 ops (`v_add_f16`, `v_cvt_f16_f32`, `v_rcp_f16` and so on) and all VOP3 opsel forms (`v_mad_u16`, `v_fma_f16`, `v_min3_i16`, `v_add_nc_u16` and so on) with `op_sel[3]` choosing which half is written (`aco_ir.cpp:691-751`, `aco_ir.cpp:607-666`). The 16 bit sources are the low half unless `op_sel` picks the high half. 16 bit shift counts use 4 bits. HLSL: without native 16 bit types use `f16tof32(x & 0xffff)` and `f32tof16`.

**G6 Constants and literals.** VOP1, VOP2 and VOPC read S0 from an SGPR, constant, VGPR or literal, and S1 only from a VGPR in the 32 bit encodings. A literal is one extra dword. GFX10 allows up to two different scalar values (SGPR, literal) in one VALU instruction.

**G7 Wave size.** GFX10 runs wave32 or wave64 per shader. Lane masks are 32 or 64 bits. Row based features (DPP, `v_permlane16`) work on rows of 16 lanes.

**G8 NaN generation.** The sign and payload bits of a NaN created by an invalid operation (0 * inf, inf minus inf, sqrt of a negative number) are not pinned down by any source I could read. Treat them as an unspecified quiet NaN, or compare against hardware.

## 5. HLSL reference helpers

The `hlsl` column uses these helpers. They all work on bit patterns because HLSL optimisers may fold `isnan` and compares under relaxed math. Compile the translated shader so that float code is strict (`precise` on every variable, DXC `-Gis` and no fast math).

```hlsl
bool nanb(float x)   { return (asuint(x) & 0x7fffffffu) > 0x7f800000u; }
bool infb(float x)   { return (asuint(x) & 0x7fffffffu) == 0x7f800000u; }
bool nanb64(double x){ uint2 u = asuint(x); uint h = u.y & 0x7fffffffu; return h > 0x7ff00000u || (h == 0x7ff00000u && u.x != 0u); }
float quiet(float x) { return asfloat(asuint(x) | 0x00400000u); }
float ftz(float x)   { uint u = asuint(x); return ((u & 0x7f800000u) == 0u) ? asfloat(u & 0x80000000u) : x; }

// v_min_f32 / v_max_f32 (quiet NaN loses against a number, negative zero is below positive zero)
float min_hw(float a, float b) {
  if (nanb(a)) return nanb(b) ? quiet(a) : b;   // which NaN wins when both are NaN: not pinned down
  if (nanb(b)) return a;
  if (a == 0.0f && b == 0.0f) return asfloat(asuint(a) | asuint(b));
  return (a < b) ? a : b; }
float max_hw(float a, float b) {
  if (nanb(a)) return nanb(b) ? quiet(a) : b;
  if (nanb(b)) return a;
  if (a == 0.0f && b == 0.0f) return asfloat(asuint(a) & asuint(b));
  return (a > b) ? a : b; }
float med3_hw(float a, float b, float c) {       // NaN case unverified
  if (nanb(a) || nanb(b) || nanb(c)) return min_hw(min_hw(a, b), c);
  float m = max_hw(max_hw(a, b), c);
  if (m == a) return max_hw(b, c);
  if (m == b) return max_hw(a, c);
  return max_hw(a, b); }

// v_fract_f32: result in [0, 1), fract(inf) = NaN, NaN stays NaN. Do not use min() for the clamp.
float fract_hw(float x) {
  if (nanb(x) || infb(x)) return asfloat(0x7fc00000u);
  float f = x - floor(x);
  return (f > asfloat(0x3f7fffffu)) ? asfloat(0x3f7fffffu) : f; }

// f32 fma. HLSL fma() is double only. The f32 product is exact in double, the sum rounds twice
// (53 bits then 24 bits), so a rare result can differ in the last bit from the hardware.
float fma_f32(float a, float b, float c) { return (float)fma((double)a, (double)b, (double)c); }

// Exact power of two scaling. HLSL ldexp(x, e) is x * exp2(e) and overflows early. This builds 2^e from bits.
float ldexp_exact(float x, int e) {
  double p = asdouble(0u, (uint)(clamp(e, -1022, 1023) + 1023) << 20);
  return (float)((double)x * p); }                // denormal result flush per MODE still has to be applied

// v_cvt_pkrtz_f16_f32: round toward zero. f32tof16 does not promise this.
uint cvt_f16_rtz(float f) {
  uint u = asuint(f), s = (u >> 16) & 0x8000u, m = u & 0x7fffffu; int e = (int)((u >> 23) & 0xffu);
  if (e == 0xff) return s | (m ? (0x7e00u | (m >> 13)) : 0x7c00u);
  e = e - 127 + 15;
  if (e >= 31) return s | 0x7bffu;               // finite overflow stays at the largest finite half
  if (e <= 0) { if (e < -10) return s; m |= 0x800000u; return s | (m >> (uint)(14 - e)); }
  return s | ((uint)e << 10) | (m >> 13); }

// v_cmp_class_f32 (class bits: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denorm, 5 -0, 6 +0, 7 +denorm, 8 +normal, 9 +inf)
bool class32(float x, uint m) {
  uint u = asuint(x), e = (u >> 23) & 0xffu, f = u & 0x7fffffu; bool s = (u >> 31) != 0u; uint c;
  if (e == 0xffu)      c = (f == 0u) ? (s ? 1u << 2 : 1u << 9) : ((f & 0x400000u) ? 1u << 1 : 1u << 0);
  else if (e == 0u)    c = (f == 0u) ? (s ? 1u << 5 : 1u << 6) : (s ? 1u << 4 : 1u << 7);
  else                 c = s ? 1u << 3 : 1u << 8;
  return (c & m & 0x3ffu) != 0u; }
// class16 and class64 are the same with 5 and 11 bit exponents (quiet bit is the top fraction bit).

// v_perm_b32 byte select, s1 supplies selectors 0..3, s0 supplies 4..7
uint perm_byte(uint s0, uint s1, uint sel) {
  if (sel >= 13u) return 0xffu;
  if (sel == 12u) return 0u;
  if (sel >= 8u) { uint src = (sel < 10u) ? s1 : s0; uint bit = (sel & 1u) ? 31u : 15u; return ((src >> bit) & 1u) ? 0xffu : 0u; }
  uint v = (sel < 4u) ? s1 : s0; return (v >> (8u * (sel & 3u))) & 0xffu; }

// v_frexp: HLSL frexp differs for inf and NaN. Denormal input handling not pinned down.
float frexp_mant32(float x) { uint u = asuint(x); uint e = (u >> 23) & 0xffu; if (e == 0xffu || (u & 0x7fffffffu) == 0u) return x; return asfloat((u & 0x807fffffu) | (126u << 23)); }
int   frexp_exp32(float x)  { uint u = asuint(x); uint e = (u >> 23) & 0xffu; if (e == 0xffu || (u & 0x7fffffffu) == 0u) return 0; return (int)e - 126; }

// float to integer with saturation and NaN to 0 (v_cvt_i32_f32 / v_cvt_u32_f32)
uint f2i_hw(float a) { return nanb(a) ? 0u : (a >= 2147483648.0f ? 0x7fffffffu : (a <= -2147483648.0f ? 0x80000000u : asuint((int)a))); }
uint f2u_hw(float a) { return (nanb(a) || a <= 0.0f) ? 0u : (a >= 4294967296.0f ? 0xffffffffu : (uint)a); }
```

Other HLSL facts used in the table:
* `min` and `max` (DXIL FMin and FMax) return the non NaN operand when exactly one operand is NaN, the same as the hardware, but do not order the zeros and may return another NaN. Never write `(a < b) ? a : b`.
* `==` is ordered (false for NaN), `!=` is unordered (true for NaN). `v_cmp_neq` equals HLSL `!=`, `v_cmp_lg` has no direct HLSL operator, write `(a < b) || (a > b)`.
* `mad(a, b, c)` may or may not be fused, `fma` is double only, so an exact f32 `v_fma_f32` is not expressible (use `fma_f32` above and accept the rare last bit difference).
* `ldexp`, `frac`, `exp2`, `log2`, `sin`, `cos`, `rcp`, `rsqrt`, `sqrt` are allowed errors in D3D and are not guaranteed to match the hardware instruction bit for bit. The hardware transcendental instructions have about 1 ULP error. Bit exact emulation of one GPU needs lookup or hardware measured tables.
* HLSL `round` is round half to even, which matches `v_rndne_*`.
* `firstbithigh(uint)` returns the index of the top set bit (31 minus clz) and 0xFFFFFFFF for 0. `firstbithigh(int)` finds the first bit that differs from the sign bit. `firstbitlow` returns the index of the lowest set bit and 0xFFFFFFFF for 0. `countbits` is popcount. `reversebits` reverses all 32 bits.
* Shifts: always mask the count explicitly (`& 31`, `& 63`, `& 15`).
* `f32tof16` rounding mode is not documented by Microsoft. Several rows assume round to nearest even and say so. Listed in section 9.
* A float to integer cast in HLSL follows LLVM (undefined) for out of range values in DXIL. Use `f2i_hw` and `f2u_hw`.

## 6. Family notes

**Min and max (G rules).** `v_min_f32`, `v_max_f32`, `v_min_f16`, `v_max_f16`, `v_min_f64`, `v_max_f64` and the three operand forms return the other operand when one operand is a quiet NaN. They order negative zero below positive zero. Signaling NaN quieting depends on MODE.IEEE and is not pinned down. LLVM models the hardware as a plain `fminnum`, and uses the older rule "comparison against NaN fails" only for the GFX6 and GFX7 legacy forms (`AMDGPUInstrInfo.td:169-173`).

**Legacy forms on GFX10.** `v_min_legacy_f32` and `v_max_legacy_f32` do not exist on GFX10 (LLVM `VOP2Instructions.td:2535-2536` has them for gfx6 and gfx7 only). The DX9 multiply survives as `v_mul_legacy_f32` (all GFX10), `v_mad_legacy_f32` and `v_mac_legacy_f32` (gfx1010 to gfx1013 only), `v_fma_legacy_f32` and `v_fmac_legacy_f32` (gfx1030 and later only). Rule for all of them: if either multiplicand is zero (any sign, and a denormal counts as zero when flushed) the product is +0.0 even when the other operand is NaN or infinity (`AMDGPUInstCombineIntrinsic.cpp:2127-2180`, `aco_optimizer.cpp:3015-3021`). For the fma form the addend is then added to +0.0, so a -0.0 addend becomes +0.0.

**Signalling compares.** The `v_cmps_*` and `v_cmpsx_*` families exist only on GFX6 and GFX7 (`VOPCInstructions.td:2375-2391`, `aco_opcodes.py:1193-1196` drops them for gfx10). GFX10 only has the plain compares, which do not change results, so no `cmps` rows exist. The sixteen float conditions are f, lt, eq, le, gt, lg, ge, o, u, nge, nlg, ngt, nle, neq, nlt, tru. The ones starting with `n` and `u` are true when either operand is NaN, the others except `o` are false in that case. `o` is true only if neither operand is NaN. Mesa uses lt, ge, eq for the ordered NIR compares and nge, nlt, neq, nlg for the unordered ones (`aco_select_nir_alu.cpp:3781-3836`).

**MAD versus FMA.** `v_mad_f32` and `v_mac_f32` are not available on RDNA2. On RDNA1 they flush denormals regardless of MODE, LLVM refuses to select them when f32 denormals are on (`AMDGPUInstrInfo.td:276-283`, `SIInstructions.td:1660-1697`). `v_fma_*` and `v_fmac_*` are fused (one rounding) and honour MODE. Mesa folds `mul+add` into `mad` or `fma` in `aco_optimizer.cpp:4480-4500`.

**Float to integer.** `v_cvt_i32_f32`, `v_cvt_u32_f32`, `v_cvt_i32_f64`, `v_cvt_u32_f64`, `v_cvt_i16_f16`, `v_cvt_u16_f16` truncate toward zero (not MODE rounding) and saturate, NaN gives 0 (knowledge `K`). `v_cvt_flr_i32_f32` floors first, `v_cvt_rpi_i32_f32` computes floor(x + 0.5). `v_cvt_off_f32_i4` returns sext(S0[3:0]) / 16. `v_cvt_f32_ubyteN` converts byte N of S0 (`AMDGPUInstrInfo.td:249-256`).

**Bit field instructions.** `v_bfe_u32` and `v_bfe_i32` use `S1[4:0]` as offset and `S2[4:0]` as width, so width 32 behaves like width 0 (`AMDGPUInstCombineIntrinsic.cpp:1499-1535`: "Hardware ignores high bits"). `v_bfi_b32` is a bitwise select with S0 as the mask. `v_bfm_b32` builds a mask with a 5 bit width. The scalar `s_bfe_*` have a 7 bit width field instead (`aco_isel_helpers.cpp:677-686`), and `s_bfm_b64` uses 6 bit width and offset (`aco_isel_helpers.cpp:666-668`).

**Find first bit.** `v_ffbh_u32` and `s_flbit_i32_b32` count leading zeros from the MSB and return 0xFFFFFFFF for zero input. `v_ffbl_b32` and `s_ff1_i32_b32` return the index of the lowest set bit and 0xFFFFFFFF for zero input. LLVM builds `ctlz` and `cttz` by clamping with `umin(result, 32)` (`AMDGPUISelLowering.cpp:3392-3410`). The signed `v_ffbh_i32` returns 0xFFFFFFFF for 0 and for -1.

**Cube instructions.** `v_cubeid_f32`, `v_cubesc_f32`, `v_cubetc_f32`, `v_cubema_f32` take the direction vector (x, y, z) in S0, S1, S2 and return face id, S, T and 2 * major axis. Mesa packs them as (tc, sc, ma, id) (`aco_select_nir_alu.cpp:2183-2196`). The selection rules are in the rows.

**Division sequence.** `v_div_scale_f32/f64`, `v_div_fmas_f32/f64`, `v_div_fixup_f32/f64/f16` only appear inside the standard divide expansion (scale, rcp, fma chain, fmas, fixup). The translator should recognise the whole sequence and emit one HLSL divide.

## 7. Instructions that look valid but are not on GFX10, or not on RDNA2

| instruction | status |
|---|---|
| `v_min_legacy_f32`, `v_max_legacy_f32` | GFX6 and GFX7 only |
| `v_cmps_*`, `v_cmpsx_*` | GFX6 and GFX7 only |
| `v_mac_f32`, `v_mad_f32`, `v_madmk_f32`, `v_madak_f32`, `v_mac_legacy_f32`, `v_mad_legacy_f32` | gfx1010 to gfx1013, error on gfx1030 (`gfx1030_err.s`) |
| `v_mul_lo_i32` | gfx1010 to gfx1013 only (`VOP3Instructions.td:2727-2729`), use `v_mul_lo_u32` |
| `v_dot4c_i32_i8`, `v_dot8c_i32_i4`, `v_dot4_*`, `v_dot8_*` | not on gfx1030 (`gfx1030_err.s`) |
| `v_fma_legacy_f32`, `v_fmac_legacy_f32` | gfx1030 and later only, no SDWA or DPP for the fmac form (`gfx1030_unsupported.s`) |
| `v_mad_f16`, `v_mac_f16`, `v_madak_f16`, `v_madmk_f16` | GFX8 and GFX9 only. GFX10 has `v_fma_f16`, `v_fmac_f16`, `v_fmamk_f16`, `v_fmaak_f16` |
| `v_add_co_u32`, `v_sub_co_u32`, `v_subrev_co_u32` as VOP2 | GFX10 has them only as VOP3 (carry to an SGPR mask). The VOP2 forms are `v_add_co_ci_u32` and friends. The plain `v_add_u32` family became `v_add_nc_u32` (no carry) |
| `v_log_clamp_f32`, `v_rcp_clamp_f32`, `v_rsq_clamp_f32` | GFX6 and GFX7 only (`VOP1Instructions.td:1533-1536`) |
| `v_cvt_pkaccum_u8_f32` | GFX6 to GFX9 only (`VOP2Instructions.td:2545,2808`) |
| DPP `wave_shl`, `wave_rol`, `wave_shr`, `wave_ror`, `row_bcast:15`, `row_bcast:31` | GFX8 and GFX9 only (`AMDGPUAsmParser.cpp:10316-10331`) |
| SDWA | GFX8 to GFX10.3 only, gone in GFX11 (`aco_validate.cpp:316-317`) |

## 8. SDWA and DPP

Both are alternative encodings of VOP1, VOP2 and VOPC. They never combine with each other and neither exists for VOP3 on GFX10 (`aco_ir.cpp:303-310`, `aco_ir.cpp:487-525`). In the `encodings` column the table lists them only where the Mesa rules allow them. Instructions with 64 bit operands or results, literal operands, `v_madmk/madak/fmamk/fmaak`, `v_readfirstlane_b32`, `v_clrexcp`, `v_swap_b32`, the mac and fmac family and `v_pk_fmac_f16` have no SDWA (and the first group no DPP).

### SDWA (sub dword addressing)

Encoding fields (`aco_assembler.cpp:1181-1226`, enum values in `SIDefines.h:1168-1201`). The SDWA dword follows the base instruction, the base `src0` field holds the code 249 and the real src0 is in the SDWA dword:

| bits | field | values |
|---|---|---|
| 7:0 | src0 | VGPR number (or SGPR or constant code on GFX9 and later, when bit 23 `s0` is clear for SGPR use) |
| 10:8 | dst_sel (VOP1, VOP2) | 0..3 BYTE_0..BYTE_3, 4 WORD_0, 5 WORD_1, 6 DWORD |
| 12:11 | dst_unused | 0 UNUSED_PAD, 1 UNUSED_SEXT, 2 UNUSED_PRESERVE |
| 13 | clamp | |
| 15:14 | omod (GFX9 and later) | |
| 18:16 | src0_sel | same encoding as dst_sel |
| 19 | src0_sext | |
| 20, 21 | src0_neg, src0_abs | |
| 23 | s0 | 1 means src0 is an SGPR or constant (GFX9 and later), 0 a VGPR |
| 26:24 | src1_sel | |
| 27 | src1_sext | |
| 29:28 | src1_abs, src1_neg | |
| 31 | s1 | 1 means src1 is an SGPR or constant, 0 a VGPR |
| VOPC only: 14:8, 15 | sdst and sd bit | destination lane mask SGPR when the bit is set, otherwise VCC. dst_sel and dst_unused do not exist for VOPC |

Reading a source: take the selected part of the 32 bit register (`BYTE_n`: bits 8n+7:8n, `WORD_n`: bits 16n+15:16n, `DWORD`: all 32 bits) and extend it to 32 bits, with sign extension if `srcN_sext` is set and zero extension otherwise. Float operations then apply `abs` and `neg` to the result (for a 16 bit float the selected word is the f16 value in the low half). The operation then runs with the extended values. A 16 bit instruction that reads a `WORD_1` gets the high half of the register as its operand.

Writing the result (not for VOPC): the result is truncated to the width of `dst_sel` and placed at that position of D: `BYTE_n` writes bits 8n+7:8n, `WORD_n` writes bits 16n+15:16n, `DWORD` writes all 32 bits. The remaining bits of D are decided by `dst_unused`: PAD sets them to 0, SEXT fills them with copies of the sign bit of the written field, PRESERVE keeps the old value of D. For a destination smaller than 32 bits ACO always encodes PRESERVE (`aco_assembler.cpp:1209`). `clamp` and `omod` act on the result before it is truncated. For VOPC, the compare result is written as a lane mask to VCC or the SGPR pair named by `sdst`.

Not pinned down: whether `abs` and `neg` are applied before or after sign extension for integer sources (they are meant for float operands), and the effect of `src_sext` on float sources. See section 9.

### DPP16 (data parallel primitives, row level)

Encoding (`aco_assembler.cpp:1029-1056`): the base instruction has src0 code 250, the DPP dword is

| bits | field |
|---|---|
| 7:0 | src0 VGPR |
| 16:8 | dpp_ctrl |
| 18 | fetch_inactive (FI, GFX10 and later) |
| 19 | bound_ctrl |
| 20, 21 | src0_neg, src0_abs |
| 22, 23 | src1_neg, src1_abs |
| 27:24 | bank_mask |
| 31:28 | row_mask |

dpp_ctrl values valid on GFX10 (`SIDefines.h:1206-1249`, `AMDGPUAsmParser.cpp:10316-10331`): 0x000 to 0x0FF `quad_perm` (two bits per lane select the source lane inside each group of four), 0x101 to 0x10F `row_shl:n`, 0x111 to 0x11F `row_shr:n`, 0x121 to 0x12F `row_ror:n` (n lanes inside a row of 16, ror rotates), 0x140 `row_mirror` (lane i reads 15 minus i), 0x141 `row_half_mirror` (lane i reads 7 minus i inside each half row), 0x150 to 0x15F `row_share:n` (every lane reads lane n of its row, GFX10 and later), 0x160 to 0x16F `row_xmask:n` (lane i reads lane i xor n inside the row, GFX10 and later). Not valid on GFX10: `wave_shl/rol/shr/ror` (0x130 to 0x13C) and `row_bcast:15/31` (0x142, 0x143).

Behaviour: DPP only changes how the first source operand (src0) is read. For lane L, `srcLane = f(dpp_ctrl, L)` inside the row of 16 lanes. D is written only for lanes with EXEC set whose row is enabled in `row_mask` (bit r for row r) and whose bank is enabled in `bank_mask` (bit b for lanes 4b to 4b+3 of every row). If the computed source lane is outside the row (`row_shl`, `row_shr` shifted out), or it is a lane with EXEC off while `fetch_inactive` is 0, the source is invalid: with `bound_ctrl = 1` src0 reads as 0 and the instruction executes, with `bound_ctrl = 0` the write for that lane is suppressed and D keeps its old value. With `fetch_inactive = 1` lanes with EXEC off are read normally. In assembler text `bound_ctrl:0` and `bound_ctrl:1` both set the bit to 1 on GFX10 (`AMDGPUAsmParser.cpp:9463`), ACO treats the bit as "read zero" (`aco_lower_to_hw_instr.cpp:876`). `abs` and `neg` of src0 and src1 apply to float operands. The second source of VOP2 is a plain VGPR.

Restrictions on GFX10 (`aco_ir.cpp:487-525`): no VOP3 form, no literals, src0 must be a VGPR, no 64 bit sources or results, compares and carry instructions must use VCC (not another SGPR), `clamp` and `omod` unavailable, and ACO avoids it for the reverse shifts and reverse subtracts (it observed the hardware applying DPP to src1 there).

### DPP8 (GFX10)

Replaces src0 code 250 by 233 (FI = 0) or 234 (FI = 1) (`aco_assembler.cpp:1063`, `SIDefines.h:1252-1257`). The DPP dword holds src0 in bits 7:0 and eight 3 bit selectors `lane_sel[0..7]` in bits 31:8 (`aco_assembler.cpp:1056-1075`). For lane L in a group of eight lanes, src0 is read from lane `lane_sel[L % 8]` of the same group. Same restrictions as DPP16 and additionally no VOP3 form. There is no row mask, bank mask or bound control: all lanes with EXEC set are written. Which value is read when the selected source lane has EXEC off and FI is 0 is not pinned down.

## 9. What could not be pinned down

Everything in the `not_pinned_down` column is open. The groups that matter most for exact results:

1. The AMD RDNA2 ISA PDF could not be read, so all `K` rows lack a direct check.
2. `s_min_*` and `s_max_*` SCC on a tie (S0 == S1).
3. `s_bfe_*` width values 33 to 127, `s_lshlN_add_u32` carry rule, `s_andn1_wrexec` and `s_andn2_wrexec` destination value, `s_movrelsd_2_b32` M0 fields, `s_flbit_i32` counting convention.
4. NaN rules: which NaN is returned when both inputs of `v_min_f32` and `v_max_f32` are NaN, signaling NaN quieting versus MODE.IEEE, `v_med3_f32` and the min3 and max3 NaN corners, the NaN result of `v_cvt_norm_*`, `v_cvt_pknorm_*`, `v_cvt_pk_u8_f32`, NaN generated by invalid operations, the NaN priority in `v_div_fixup_*`.
5. Denormal behaviour of `v_frexp_*`, `v_rsq_f32`, `v_sqrt_f32`, `v_cmp_class_*`, `v_ldexp_f32` and the exact flush rule of `v_mul_legacy_f32` when denormals are kept.
6. Accuracy and error patterns of the approximating instructions (`v_rcp_*`, `v_rsq_*`, `v_sqrt_f64`, `v_log_f32`, `v_exp_f32`, `v_sin_*`, `v_cos_*`) and the valid input range of sin and cos (the ISA text says 256 revolutions, LLVM first reduces with fract).
7. Rounding of the scale in `v_cvt_norm_*` and `v_cvt_pknorm_*`, the float to byte rounding in `v_cvt_pk_u8_f32`, whether `v_cvt_rpi_i32_f32` adds 0.5 exactly, the upper half of `v_sat_pk_u8_i16`.
8. `v_perm_b32` selector values 8 to 11 and 13 and above, `v_lerp_u8` lane truncation, `v_sad_hi_u8` shift, `v_msad_u8` mask side, `v_mad_i64_i32` carry out, `v_mbcnt_*` in wave32, `v_permlane16/x16` inactive lane rules.
9. `v_div_scale_f32/f64` (the exponent thresholds and flag rule) and the exact scale factors of `v_div_fmas_f32/f64` (2^32 and 2^64).
10. No semantics at all (not seen in the sources and not remembered reliably): `v_mullit_f32`, `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`, `v_mqsad_u32_u8`, `v_trig_preop_f64`, `v_swaprel_b32`, `v_interp_p1ll_f16`, `v_interp_p1lv_f16`, `v_interp_p2_f16`, `s_subvector_loop_begin`, `s_subvector_loop_end`.
11. SDWA: order of source selection, sign extension and abs or neg, effect of `src_sext` on float operands. DPP8 with FI = 0 reading an inactive lane.
12. HLSL side: the rounding mode of `f32tof16` (the f16 emulations assume round to nearest even), and whether the DXIL float to int conversions saturate (the table writes the clamps by hand).
