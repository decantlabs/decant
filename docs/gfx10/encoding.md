# GFX10 (RDNA1 and RDNA2) shader instruction encoding

Scope: AMD GFX10 as used by RDNA1 (10.1) and RDNA2 (10.3, the Xbox Series class of GPU). Wave32 and wave64.
Companion table: `opcodes.tsv` in this folder. Check tool: `aco_opcode_crosscheck` (crate `decant-shader-tools`).

Clean room statement. Every fact below comes from one of these three source families and is cited in place:

* Mesa ACO, MIT: `src/amd/compiler/aco_assembler.cpp`, `aco_ir.h`, `aco_opcodes.py` (written as `mesa:file:line`)
* LLVM AMDGPU backend, Apache 2 with LLVM exception: `llvm/lib/Target/AMDGPU/*.td`, `SIDefines.h`, `Disassembler/AMDGPUDisassembler.cpp` (written as `llvm:file:line`)
* LLVM tests: `llvm/test/MC/AMDGPU` and `llvm/test/MC/Disassembler/AMDGPU` (written as `llvmtest:file:line`)

Local checkouts used: Mesa (`ref-mesa`) and LLVM (`ref-llvm`). No console emulator or translator project was opened.
Line numbers refer to the checkouts made on 2026 11 24 (Mesa commit f8281cfa33e2e752d37b0d619ca53111bbe765de).

Bit ranges are written `hi:lo` and are counted inside the named dword. A 64 bit instruction is dword0 (bits 31:0) followed by dword1 (bits 63:32 in LLVM numbering). Dwords are little endian in memory.

## 1. Format selection and instruction length

The format is chosen from the top bits of dword0. Prefixes below are taken from the `Inst{...}` assignments in the LLVM .td files and agree with the constants in `mesa:aco_assembler.cpp`.

```
31:23 == 0x17d  SOP1   (llvm:SOPInstructions.td:80,  mesa:aco_assembler.cpp:201)
31:23 == 0x17e  SOPC   (llvm:SOPInstructions.td:1350, mesa:aco_assembler.cpp:213)
31:23 == 0x17f  SOPP   (llvm:SOPInstructions.td:1547, mesa:aco_assembler.cpp:227)
31:28 == 0xb    SOPK   (llvm:SOPInstructions.td:1082, mesa:aco_assembler.cpp:185)   test after the three above
31:30 == 0b10   SOP2   (llvm:SOPInstructions.td:635,  mesa:aco_assembler.cpp:155)   test after the four above
31    == 0      VOP family: 31:25 == 0x3e VOPC, 31:25 == 0x3f VOP1, otherwise VOP2
                (llvm:VOP1Instructions.td:14, VOP2Instructions.td:13, VOPCInstructions.td:13)
31:26 == 0x32   VINTRP (llvm:SIInstrFormats.td:341)
31:24 == 0xcc   VOP3P  (llvm:VOPInstructions.td:659, mesa:aco_assembler.cpp:1144)
31:26 == 0x35   VOP3   (llvm:VOPInstructions.td:300, mesa:aco_assembler.cpp:1076)
31:26 == 0x36   DS     (llvm:DSInstructions.td:1477)
31:26 == 0x37   FLAT, SCRATCH and GLOBAL (llvm:FLATInstructions.td:164), segment in 15:14
31:26 == 0x38   MUBUF  (llvm:BUFInstructions.td:2582)
31:26 == 0x3a   MTBUF  (llvm:BUFInstructions.td:3244)
31:26 == 0x3c   MIMG   (llvm:SIInstrFormats.td:355 region)
31:26 == 0x3d   SMEM   (llvm:SMInstructions.td:1235)
31:26 == 0x3e   EXP    (llvm:SIInstrFormats.td:508 region)
```

Facts about the table above that are confirmed by running the decode on 38157 LLVM disassembler vectors with no unknown format (see `aco_opcode_crosscheck`).

Length of one instruction in dwords:

* SOP1, SOP2, SOPC, SOPK, SOPP, VOP1, VOP2, VOPC, VINTRP (native): 1 dword.
* VOP3, VOP3P, SMEM, DS, MUBUF, MTBUF, MIMG, FLAT, GLOBAL, SCRATCH, EXP: 2 dwords.
* Plus one literal dword when a source field holds 255 (section 3).
* Plus one mandatory constant dword for a few instructions: VOP2 v_madmk_f32, v_madak_f32, v_fmamk_f32, v_fmaak_f32 (llvm:VOP2Instructions.td:25 class VOP2_MADKe, imm in 63:32), and SOPK s_setreg_imm32_b32 (llvm:SOPInstructions.td:1085 class SOPK_Real64). The `mandatory_literal` column of `opcodes.tsv` marks these, derived from the vector byte lengths.
* VOP1, VOP2 and VOPC become 2 dwords when src0 is 249 (SDWA) or 250 (DPP16) or 233 and 234 (DPP8). The extra dword follows dword0 directly (section 5 and 6). A literal then still comes last.
* MIMG gets 0 to 3 extra NSA (non sequential address) dwords. The count is in dword0 bits 2:1 (llvm:SIInstrFormats.td:404 MIMGe_gfx10 `nsa`, mesa:aco_assembler.cpp:68 get_mimg_nsa_dwords and :783 `encoding |= nsa_dwords << 1`). NSA dwords come after dword1. Each holds four 8 bit VGPR numbers (mesa:aco_assembler.cpp:811 to 818). The first address register is in dword1 bits 7:0, the following ones are packed in the NSA dwords.
* Mesa appends the literal after everything else (mesa:aco_assembler.cpp:1430).
* LLVM does not decide the length from a single rule. It tries the 64 bit decoder tables first and then the 32 bit tables and takes the length from the table that matches (llvm:Disassembler/AMDGPUDisassembler.cpp:741 to 960, comment at :749 "better to switch encoding length using some bit predicate but it is unknown yet, so try all we can"). A walker for GFX10 should therefore use the table above for the base length and then add the extras listed here.

Instruction stream facts that follow: there is no 16 bit or 48 bit instruction on GFX10, so every length is a multiple of 4 bytes.

## 2. Operand source encoding (9 bits, values 0 to 511)

Used by the 9 bit source fields of VOP1, VOP2 (src0), VOPC (src0), VOP3 and VOP3P. The 8 bit scalar source fields of SOP1, SOP2, SOPC use values 0 to 255 of the same table (no VGPRs). Scalar destinations (SDST, 7 bits) use values 0 to 127 of the same table.

```
0 to 105    SGPR s0 to s105          (llvm:SIDefines.h:1196 SRC_SGPR_MAX_GFX10 = 361 means 256 + 105 in the SDWA source encoding)
106         vcc_lo                   (llvm:SIRegisterInfo.td:179, mesa:aco_ir.h:440)
107         vcc_hi                   (llvm:SIRegisterInfo.td:181)
108 to 123  ttmp0 to ttmp15          (llvm:SIRegisterInfo.td:335 gfx9plus = 108 + index)
124         m0                       (llvm:SIRegisterInfo.td:227 M0_gfxpre11 = 124)
125         null                     (llvm:SIRegisterInfo.td:231 SGPR_NULL_gfxpre11 = 125)
126         exec_lo                  (llvm:SIRegisterInfo.td:206)
127         exec_hi                  (llvm:SIRegisterInfo.td:207)
128         inline integer 0
129 to 192  inline integers 1 to 64  (value = code minus 128)   (mesa:aco_ir.h:524 and :562)
193 to 208 inline integers minus 1 to minus 16 (value = 192 minus code)   (mesa:aco_ir.h:526)
209 to 232  not used on GFX10 in the sources read (230 and 231 are src_flat_scratch_base_lo and hi, llvm:SIRegisterInfo.td:273 and :274, a later generation)
233         DPP8 marker, fetch inactive 0     (llvm:SIDefines.h:1255 DPP8_FI_0 = 0xE9, mesa:aco_assembler.cpp:1064)
234         DPP8 marker, fetch inactive 1     (llvm:SIDefines.h:1256 DPP8_FI_1 = 0xEA)
235         src_shared_base                   (llvm:SIRegisterInfo.td:267)
236         src_shared_limit                  (llvm:SIRegisterInfo.td:268)
237         src_private_base                  (llvm:SIRegisterInfo.td:269)
238         src_private_limit                 (llvm:SIRegisterInfo.td:270)
239         src_pops_exiting_wave_id          (llvm:SIRegisterInfo.td:288, mesa:aco_ir.h:463 "GFX9 to GFX10.3", llvmtest:gfx11_asm_operands.s:115 prints 0xef for GFX10)
240         0.5       241  minus 0.5        (mesa:aco_ir.h:528 and :530)
242         1.0       243  minus 1.0
244         2.0       245  minus 2.0
246         4.0       247  minus 4.0
248         1/(2*pi) = 0.15915494 (single precision 0x3e22f983), GFX8 and newer   (mesa:aco_ir.h:675 to 678)
249         SDWA marker, real operand is in the SDWA dword   (mesa:aco_assembler.cpp:1189, llvmtest:gfx10_vop2_sdwa.txt:7 byte 0xf9)
250         DPP16 marker, real operand is in the DPP dword   (mesa:aco_assembler.cpp:1035)
251         src_vccz                           (llvm:SIRegisterInfo.td:217, llvmtest:gfx11_asm_operands.s:59 prints 0xfb)
252         src_execz                          (llvm:SIRegisterInfo.td:218, llvmtest:gfx11_asm_operands.s:19 prints 0xfc)
253         src_scc                            (llvm:SIRegisterInfo.td:219, mesa:aco_ir.h:464)
254         src_lds_direct                     (llvm:SIRegisterInfo.td:297, llvmtest:gfx11_asm_operands.s:95 prints 0xfe)
255         32 bit literal constant follows the instruction
256 to 511  VGPR v0 to v255                    (llvm:SIDefines.h:1187 SRC_SGPR_MASK = 0x100)
```

Important GFX10 details:

* m0 and null are swapped relative to GFX11. On GFX10 m0 is 124 and null is 125. Mesa keeps GFX11 numbers internally and swaps while assembling for GFX10 and older (mesa:aco_ir.h:458 to 459, mesa:aco_assembler.cpp:99 to 108). LLVM test `llvmtest:gfx10-null-reg.txt` shows null as byte 0x7d. LLVM test `llvmtest:gfx10_asm_sop.s:31` shows `s_abs_i32 m0, s1` with sdst byte 0xfc.
* The wave size decides the meaning of lane mask operands. In wave32 a lane mask is one SGPR (vcc_lo, exec_lo). In wave64 it is a pair (vcc, exec). Same encodings 106, 126 (llvmtest:gfx10_vopc.txt prints `vcc_lo` or `vcc` for identical bytes).
* Inline constants have the width of the operand. A 16 bit float operand reads code 240 as 0.5 in half precision, a 64 bit float operand reads it as 0.5 in double precision (mesa:aco_ir.h:516 c16, :553 c64, :779 constantValue64).
* GFX10 can read two different scalar values (SGPR, literal, m0, vcc and so on) in one VALU instruction, except 64 bit shifts which allow one. Implicit vcc reads count. null is free. (llvmtest:gfx10-constant-bus.s:6 to 70.) This is an assembler rule, the bytes do not change.
* src_lds_direct is accepted only by some instructions (llvmtest:gfx10_err_pos.s:943 shows it is rejected for 16 bit ALU forms). The check by the hardware is not covered by these sources.
* Not confirmed for GFX10 hardware in the sources read: codes 235 to 238 (the encodings are defined for the target family in llvm SIRegisterInfo.td, no GFX10 vector exercises them).

## 3. The 32 bit literal rule

* When any source field of an instruction equals 255 the instruction is followed by one 32 bit dword with the constant (mesa:aco_assembler.cpp:1430 to 1437 appends the first literal operand only, so there is at most one literal dword per instruction).
* The same literal value may be named in several source fields, they share the one dword (llvmtest:gfx10-constant-bus.s:50, `v_div_fmas_f64 ... 0x12345678, 0x12345678, 0x12345678` is 12 bytes in total).
* For 64 bit operands the literal is not 64 bit. Float64 operands treat the dword as the HIGH 32 bits of the double, low bits zero (llvm:Disassembler/AMDGPUDisassembler.cpp:1869 to 1873, `Val <<= 32`). Integer 64 operands treat it as the low 32 bits, zero extended unless Mesa recorded a sign extension (llvm:AMDGPUDisassembler.cpp:1875 to 1878, mesa:aco_ir.h:795 `signext`).
* GFX10 allows a literal in VOP3 and VOP3P (llvmtest:gfx10-vop3-literal.txt, llvmtest:gfx10_vop3p_literalv216.txt). For VOP3P the meaning of the literal halves was not read.
* In VOP3 the literal dword comes after dword1 (12 byte instruction), see llvmtest:gfx10-vop3-literal.txt:5.
* For VOP2 forms with a trailing mandatory constant (madmk, madak, fmamk, fmaak) the constant dword is always present and is NOT introduced by 255 in a source field (llvm:VOP2Instructions.td:25).
* SOP1, SOP2 and SOPC accept 255 in a source field and then take one literal dword. SOPK s_setreg_imm32_b32 is the one SOPK form with a second dword (llvm:SOPInstructions.td:1085 SOPK_Real64, :2763 SOPK_Real64_gfx10). SOP2_Real64 exists only for GFX11 and newer (llvm:SOPInstructions.td:2618 to 2628).
* A literal combined with SDWA or DPP words: not confirmed from the sources read.

## 4. Field layouts per format

Source of each layout is the `Inst{hi-lo}` list of the named LLVM class, cross read against the Mesa emit function. Where both are given they agree unless a remark says otherwise.

SOP1 (llvm:SOPInstructions.td:77 to 80, mesa:aco_assembler.cpp:197)
```
7:0   SSRC0   (source encoding, section 2, 255 = literal)
15:8  OP
22:16 SDST
31:23 0x17d
```

SOP2 (llvm:SOPInstructions.td:631 to 635, mesa:aco_assembler.cpp:151)
```
7:0   SSRC0
15:8  SSRC1
22:16 SDST
29:23 OP (7 bits)
31:30 0b10
```

SOPK (llvm:SOPInstructions.td:1079 to 1082, mesa:aco_assembler.cpp:164)
```
15:0  SIMM16
22:16 SDST (also the source register for instructions that read one)
27:23 OP (5 bits)
31:28 0xb
```
s_setreg_imm32_b32 has a second dword with a 32 bit immediate.

SOPC (llvm:SOPInstructions.td:1347 to 1350, mesa:aco_assembler.cpp:209)
```
7:0   SSRC0
15:8  SSRC1
22:16 OP (7 bits)
31:23 0x17e
```

SOPP (llvm:SOPInstructions.td:1545 to 1547, mesa:aco_assembler.cpp:221)
```
15:0  SIMM16 (for branches: signed distance in dwords from the NEXT instruction, mesa:aco_assembler.cpp:1699 `target.offset - branch.pos - 1`)
22:16 OP (7 bits)
31:23 0x17f
```

SMEM, GFX10 (llvm:SMInstructions.td:1232 to 1241 and :1248 to 1249, mesa:aco_assembler.cpp:240 to 343)
```
dword0
 5:0    SBASE   (SGPR pair number divided by 2)
 12:6   SDATA   (destination for loads, source for stores, 7 bit scalar register code)
 14     DLC
 16     GLC
 25:18  OP
 31:26  0x3d
dword1 (bits 63:32 in llvm numbering)
 20:0   OFFSET  (21 bit byte offset)
 24:21  unused
 31:25  SOFFSET (scalar register code, set to 125 = null when no register offset is used, mesa:aco_assembler.cpp:307)
```
Note: bit 17 is not an IMM flag on GFX10, an immediate offset is signalled by SOFFSET = null (mesa:aco_assembler.cpp:305 to 316).

VOP1 (llvm:VOP1Instructions.td:14 to 20, mesa:aco_assembler.cpp:363)
```
8:0    SRC0 (section 2, 249 SDWA, 250 DPP16, 233 or 234 DPP8, 255 literal)
16:9   OP
24:17  VDST
31:25  0x3f
```

VOP2 (llvm:VOP2Instructions.td:13 to 19, mesa:aco_assembler.cpp:346)
```
8:0    SRC0
16:9   VSRC1 (VGPR number, 8 bits)
24:17  VDST
30:25  OP
31     0
```
Instructions that read vcc implicitly (v_cndmask_b32, v_add_co_ci_u32 and friends) use vcc with no field for it. In VOP3 form the lane mask is an explicit SGPR.

VOPC (llvm:VOPCInstructions.td:13 to 19, mesa:aco_assembler.cpp:382)
```
8:0    SRC0
16:9   VSRC1
24:17  OP
31:25  0x3e
```
The result goes to vcc (v_cmp) or exec (v_cmpx). With SDWA it can go to an SGPR (section 5).

VINTRP native (llvm:SIInstrFormats.td:341 to 352, mesa:aco_assembler.cpp:441 to 451)
```
7:0    VSRC (the i or j VGPR, or the parameter selector for v_interp_mov_f32 in 1:0)
9:8    ATTRCHAN
15:10  ATTR
17:16  OP
25:18  VDST
31:26  0x32
```
VOP3 form of the same instructions: opcode 0x200 + op (llvmtest:gfx10_vop3.txt:7488, `v_interp_p1_f32_e64`, llvm:VOP3Instructions.td:2754 VOP3Interp_Real_gfx10<0x200>). The 16 bit interpolation instructions (v_interp_p1ll_f16 and so on) exist only as VOP3.

VOP3A (llvm:VOPInstructions.td:236 to 261 and :300 to 303, :318 to 321, mesa:aco_assembler.cpp:1076 to 1141)
```
dword0
 7:0    VDST
 10:8   ABS     (bit 8 src0, 9 src1, 10 src2)
 14:11  OP_SEL  (bit 11 src0, 12 src1, 13 src2, 14 destination, used by the 16 bit forms)
 15     CLAMP
 25:16  OP (10 bits)
 31:26  0x35
dword1
 8:0    SRC0
 17:9   SRC1
 26:18  SRC2
 28:27  OMOD   (0 none, 1 times 2, 2 times 4, 3 divide by 2)
 31:29  NEG    (bit 29 src0, 30 src1, 31 src2)
```
Unused source fields are written as inline constant 0 (code 128) by Mesa on GFX10 (mesa:aco_assembler.cpp:1119 to 1126). OP_SEL placement is from mesa:aco_assembler.cpp:1099 (`vop3.opsel << 11`). The OMOD value meaning is the standard output modifier (llvmtest vectors `mul:2`, `mul:4`, `div:2` at gfx10_vop3.txt near `v_interp_mov_f32_e64` encode 0x08, 0x10, 0x18 in the top byte, so 1 is times 2, 2 is times 4, 3 is divide by 2).

VOP3B (llvm:VOPInstructions.td:456 to 477 and :670 to 675, mesa:aco_assembler.cpp:1108 to 1110)
Same as VOP3A except dword0 bits 14:8 are SDST (7 bit scalar register code, the carry out or lane mask result) instead of ABS and OP_SEL. CLAMP stays at bit 15. Used by v_add_co_u32, v_sub_co_u32, v_subrev_co_u32, the e64 forms of v_add_co_ci_u32, v_sub_co_ci_u32, v_subrev_co_ci_u32, v_div_scale_f32 and f64, v_mad_u64_u32, v_mad_i64_i32.

VOP3 form of VOPC: VDST (7:0) carries the SGPR destination of the compare (llvmtest:gfx10_vop3c.txt `v_cmp_lt_f32_e64 s10, -1, v2` has byte 0x0a). For v_cmpx the VDST byte is 0x7e (exec_lo) and no destination is printed (llvmtest:gfx10_vop3cx.txt, first lines).

Opcode ranges inside the 10 bit VOP3 opcode field on GFX10 (llvm *_Real_gfx10 multiclasses, mesa:aco_assembler.cpp:1081 to 1090):
```
0x000 to 0x0ff  VOPC instructions in VOP3 form        (op unchanged)
0x100 to 0x13f  VOP2 instructions in VOP3 form        (0x100 + VOP2 op)
0x140 to 0x17f  instructions that exist only in VOP3  (for example v_mad_f32 0x141)
0x180 to 0x1ff  VOP1 instructions in VOP3 form        (0x180 + VOP1 op)
0x200 to 0x202  VOP3 form of v_interp_p1_f32, v_interp_p2_f32, v_interp_mov_f32
0x203 to 0x377  instructions that exist only in VOP3
```
The `vop3_opcode_hex` column of `opcodes.tsv` lists the VOP3 opcode for every VOP1, VOP2, VOPC and VINTRP row. Mesa uses 0x140 for VOP1 on GFX8 and GFX9 and 0x180 on GFX10 and newer (mesa:aco_assembler.cpp:1085 to 1089), and 0x270 for VINTRP on GFX8 and GFX9 (not GFX10, the LLVM vectors use 0x200).

VOP3P (llvm:VOPInstructions.td:479 to 536 and :659 to 662, mesa:aco_assembler.cpp:1144 to 1178)
```
dword0
 7:0    VDST
 10:8   NEG_HI  (bit 8 src0, 9 src1, 10 src2)
 13:11  OP_SEL  (low half select, bits 11, 12, 13 for src0, src1, src2)
 14     OP_SEL_HI bit for src2
 15     CLAMP
 23:16  OP (8 bits)
 31:24  0xcc
dword1
 8:0    SRC0
 17:9   SRC1
 26:18  SRC2
 28:27  OP_SEL_HI bits for src0 (bit 27) and src1 (bit 28)
 31:29  NEG (low half negate, bit 29 src0, 30 src1, 31 src2)
```

DS (llvm:DSInstructions.td:1473 to 1481 base class used for GFX6, GFX7, GFX10 and GFX11, mesa:aco_assembler.cpp:503 to 534)
```
dword0
 7:0    OFFSET0
 15:8   OFFSET1
 16     unused
 17     GDS
 25:18  OP (8 bits)
 31:26  0x36
dword1
 7:0    ADDR  (VGPR)
 15:8   DATA0 (VGPR)
 23:16  DATA1 (VGPR)
 31:24  VDST  (VGPR)
```
Instructions with a single 16 bit offset use OFFSET1:OFFSET0 as one 16 bit field. M0 is read implicitly by some DS instructions on older chips. The sources read do not state whether GFX10 requires it (see list of unconfirmed items).

MUBUF (llvm:BUFInstructions.td:2581 to 2586 and :2606 to 2625, mesa:aco_assembler.cpp:554 to 603)
```
dword0
 11:0   OFFSET
 12     OFFEN
 13     IDXEN
 14     GLC
 15     DLC
 16     LDS
 24:18  OP low 7 bits
 25     OP bit 7
 31:26  0x38
dword1
 7:0    VADDR
 15:8   VDATA
 20:16  SRSRC (SGPR quad number divided by 4)
 21     unused
 22     SLC
 23     TFE
 31:24  SOFFSET (scalar register code or 128 for inline constant 0)
```

MTBUF (llvm:BUFInstructions.td:3242 to 3248, :3268 to 3273 and :3360 to 3367, mesa:aco_assembler.cpp:643 to 697)
```
dword0
 11:0   OFFSET
 12     OFFEN
 13     IDXEN
 14     GLC
 15     DLC
 18:16  OP low 3 bits
 25:19  FORMAT (7 bit unified data and number format, GFX10)
 31:26  0x3a
dword1
 7:0    VADDR
 15:8   VDATA
 20:16  SRSRC
 21     OP bit 3
 22     SLC
 23     TFE
 31:24  SOFFSET
```

MIMG (llvm:SIInstrFormats.td:355 to 376 and :404 to 418, mesa:aco_assembler.cpp:737 to 824)
```
dword0
 0      OP bit 7
 2:1    NSA (number of extra address dwords after dword1)
 5:3    DIM (dimension code)
 7      DLC
 11:8   DMASK
 12     UNORM
 13     GLC
 14     unused on GFX10 (was DA before)
 15     R128
 16     TFE
 17     LWE
 24:18  OP low 7 bits
 25     SLC
 31:26  0x3c
dword1
 7:0    VADDR (first address register)
 15:8   VDATA
 20:16  SRSRC (SGPR octuple or quad number divided by 4)
 25:21  SSAMP (SGPR quad number divided by 4)
 29:26  unused
 30     A16
 31     D16
```

FLAT, GLOBAL, SCRATCH (llvm:FLATInstructions.td:2969 to 2977 for GFX10 and :157 to 171 for the base, mesa:aco_assembler.cpp:883 to 947)
```
dword0
 11:0   OFFSET (signed 12 bit for GLOBAL and SCRATCH. For plain FLAT the field must be 0 because of a hardware bug, mesa:aco_assembler.cpp:911 to 918)
 12     DLC
 13     LDS
 15:14  SEG (0 FLAT, 1 SCRATCH, 2 GLOBAL)
 16     GLC
 17     SLC
 24:18  OP
 31:26  0x37
dword1
 7:0    ADDR  (VGPR)
 15:8   DATA  (VGPR)
 22:16  SADDR (scalar register code, 125 = null means "no SGPR base", 127 for scratch without any address, mesa:aco_assembler.cpp:935 to 943)
 23     NV (0 on GFX10)
 31:24  VDST
```

EXP (llvm:SIInstrFormats.td:508 to 541, mesa:aco_assembler.cpp:989 to 1021)
```
dword0
 3:0    EN (component enable mask, with COMPR it is 2 bits per pair)
 9:4    TARGET
 10     COMPR
 11     DONE
 12     VM (valid mask)
 31:26  0x3e
dword1
 7:0    VSRC0
 15:8   VSRC1
 23:16  VSRC2
 31:24  VSRC3
```
Export targets on GFX10 (llvm:SIDefines.h:1261 to 1289): MRT0 to MRT7 are 0 to 7, MRTZ 8, NULL 9, POS0 to POS3 are 12 to 15, POS4 16 (GFX10 and newer), PRIM 20 (GFX10 and newer), PARAM0 to PARAM31 are 32 to 63.

## 5. SDWA word (VOP1, VOP2, VOPC only)

Dword0 is the normal VOP1, VOP2 or VOPC dword with SRC0 = 249. A second dword follows (llvm:VOPInstructions.td:701 to 716 and :740 to 775, mesa:aco_assembler.cpp:1181 to 1231, llvmtest:gfx10_vop2_sdwa.txt:7):
```
7:0    SRC0 VGPR number (or SGPR number when bit 23 is set)
10:8   DST_SEL      0 byte0, 1 byte1, 2 byte2, 3 byte3, 4 word0, 5 word1, 6 dword
12:11  DST_UNUSED   0 pad with zero, 1 sign extend, 2 preserve (llvm:SIDefines.h:1180)
13     CLAMP
15:14  OMOD
18:16  SRC0_SEL     same codes as DST_SEL
19     SRC0_SEXT
20     SRC0_NEG
21     SRC0_ABS
22     unused
23     S0           1 means SRC0 is an SGPR
26:24  SRC1_SEL
27     SRC1_SEXT
28     SRC1_NEG
29     SRC1_ABS
30     unused
31     S1           1 means SRC1 is an SGPR
```
Variant for VOPC (llvm:VOPInstructions.td:760 to 765 class VOP_SDWA9Be, mesa:aco_assembler.cpp:1195 to 1203): bits 14:8 become SDST (SGPR destination of the compare) and bit 15 is SD, which selects SDST when 1 and vcc when 0. OMOD and DST_SEL and DST_UNUSED do not exist. CLAMP stays at 13.
The `has_sdwa` flag in `opcodes.tsv` is 1 when a GFX10 test vector with SDWA exists for the mnemonic. The sources read give no complete list, so a 0 means "no test vector found", see the unconfirmed list.

## 6. DPP words (VOP1, VOP2, VOPC only)

DPP16 (llvm:VOPInstructions.td:883 to 903, mesa:aco_assembler.cpp:1028 to 1053, llvmtest:gfx10_vop1_dpp16.txt:2851). Dword0 has SRC0 = 250. The second dword is:
```
7:0    SRC0 VGPR number
16:8   DPP_CTRL
17     unused
18     FI (fetch inactive)
19     BOUND_CTRL
20     SRC0_NEG
21     SRC0_ABS
22     SRC1_NEG
23     SRC1_ABS
27:24  BANK_MASK
31:28  ROW_MASK
```
DPP_CTRL values seen in GFX10 vectors (llvmtest:gfx10_asm_dpp16.s, gfx10_vop*_dpp16.txt), layout from llvm:SIDefines.h:1206 to 1250:
```
0x000 to 0x0ff  quad_perm (two bits per lane, lane 0 in bits 1:0)
0x101 to 0x10f  row_shl 1 to 15
0x111 to 0x11f  row_shr 1 to 15
0x121 to 0x12f  row_ror 1 to 15
0x140           row_mirror
0x141           row_half_mirror
0x150 to 0x15f  row_share 0 to 15   (GFX10 and newer)
0x160 to 0x16f  row_xmask 0 to 15   (GFX10 and newer)
```
wave_shl1 (0x130), wave_rol1 (0x134), wave_shr1 (0x138), wave_ror1 (0x13c), row_bcast15 (0x142) and row_bcast31 (0x143) exist in the shared enum but no GFX10 vector uses them and the assembler test rejects `row_bcast` (llvmtest:gfx10_err_pos.s:1018). Treat them as not available on GFX10.

DPP8 (llvm:VOPInstructions.td:1196 to 1201, mesa:aco_assembler.cpp:1056 to 1073, llvmtest:gfx10_vop1_dpp8.txt:6 and :9). Dword0 has SRC0 = 233 (fetch inactive 0) or 234 (fetch inactive 1). The second dword is:
```
7:0    SRC0 VGPR number
10:8   lane select for lane 0 of each group of 8
13:11  lane select for lane 1
16:14  lane 2
19:17  lane 3
22:20  lane 4
25:23  lane 5
28:26  lane 6
31:29  lane 7
```
(LLVM: `Inst{63-40} = dpp8{23-0}`, Mesa: `encoding |= dpp.lane_sel << 8`. The identity vector `dpp8:[0,1,2,3,4,5,6,7]` encodes bytes 0x88 0xc6 0xfa, which read as eight 3 bit groups from the lowest bit give 0,1,2,3,4,5,6,7. So lane 0 sits in the lowest field. Source: llvmtest:gfx10_vop1_dpp8.txt:6.)

DPP and SDWA are not available for VOP3 or VOP3P on GFX10 (no GFX10 vector has them, llvmtest:gfx10_unsupported_e64_dpp.s is an error test file).

## 7. Which instructions have which forms

`opcodes.tsv` has these related columns:

* `format` and `opcode_dec`/`opcode_hex`: native format and the opcode number in that format (the VOP3 opcode for VOP3 only instructions).
* `vop3_opcode_hex` and `vop3_opcode_basis`: opcode of the VOP3 encoding of a VOP1, VOP2, VOPC or VINTRP instruction. `observed` means an LLVM e64 vector shows it. `rule_unverified` means the number was computed from the ranges in section 4 and no e64 vector exists for the mnemonic. It is empty for the VOP2 forms that carry a mandatory constant.
* `vop3_variant`: VOP3A, VOP3B, or the two VOPC remarks.
* `has_e64`, `has_sdwa`, `has_dpp16`, `has_dpp8`: set when a GFX10 test vector of that kind exists.
* `size_dwords`, `mandatory_literal`: length rules from section 1.

## 8. Gaps and cautions

See the final answer of the task and the `notes` column of `opcodes.tsv`. In short: Mesa propagates older generation opcode numbers into GFX10 unless it marks an opcode as removed, so a Mesa row alone is not proof the instruction exists on GFX10. The column `sources` says which of the three sources (mesa, llvm_vectors, llvm_td) agree, and `status` is `confirmed` when at least two agree.
