//go:build arm64

#include "textflag.h"

// Prompt-reading (GEMM) micro-kernels: 2 weight rows x 4 activation rows per
// call, activations in the q8k4 layout (see gemm.go). Instructions the Go
// assembler doesn't know are hand-encoded; arguments are register numbers.
#define ZERO(d) WORD $(0x6F00E400 | (d))                                        // MOVI Vd.2D, #0
#define SDOT(m, n, d) WORD $(0x4E809400 | ((m)<<16) | ((n)<<5) | (d))          // Vd.4S += Vn.16B . Vm.16B
#define SCVTF4S(n, d) WORD $(0x4E21D800 | ((n)<<5) | (d))                       // Vd.4S = float(Vn.4S)
#define ADDP4S(m, n, d) WORD $(0x4EA0BC00 | ((m)<<16) | ((n)<<5) | (d))         // pairwise add of Vn:Vm
#define UXTL8B(n, d) WORD $(0x2F08A400 | ((n)<<5) | (d))                        // Vd.8H = Vn.8B
#define UXTL2_16B(n, d) WORD $(0x6F08A400 | ((n)<<5) | (d))                     // Vd.8H = Vn.16B[8:]
#define UXTL4H(n, d) WORD $(0x2F10A400 | ((n)<<5) | (d))                        // Vd.4S = Vn.4H
#define UXTL2_8H(n, d) WORD $(0x6F10A400 | ((n)<<5) | (d))                      // Vd.4S = Vn.8H[4:]
#define FCVTL4H(n, d) WORD $(0x0E217800 | ((n)<<5) | (d))                       // Vd.4S = float32(Vn.4H f16)
#define ZIP1S(m, n, d) WORD $(0x4E803800 | ((m)<<16) | ((n)<<5) | (d))          // Vd.4S = [n0 m0 n1 m1]
#define ZIP2S(m, n, d) WORD $(0x4E807800 | ((m)<<16) | ((n)<<5) | (d))          // Vd.4S = [n2 m2 n3 m3]
#define MLA4S(m, n, d) WORD $(0x4EA09400 | ((m)<<16) | ((n)<<5) | (d))          // Vd.4S += Vn.4S * Vm.4S
#define SUB4S(m, n, d) WORD $(0x6EA08400 | ((m)<<16) | ((n)<<5) | (d))          // Vd.4S = Vn.4S - Vm.4S
#define EXT8(m, n, d) WORD $(0x6E004000 | ((m)<<16) | ((n)<<5) | (d))           // Vd = bytes 8..23 of Vn:Vm
#define DUPD(n, i, d) WORD $(0x4E080400 | ((i)<<20) | ((n)<<5) | (d))           // Vd.2D = Vn.D[i]
// by-element forms; i is the lane, m < 16 for H lanes
#define SMULLH(m, i, n, d) WORD $(0x0F40A000 | ((((i)>>1)&1)<<21) | (((i)&1)<<20) | ((m)<<16) | (((i)>>2)<<11) | ((n)<<5) | (d))  // Vd.4S = Vn.4H * Vm.H[i]
#define SMLALH(m, i, n, d) WORD $(0x0F402000 | ((((i)>>1)&1)<<21) | (((i)&1)<<20) | ((m)<<16) | (((i)>>2)<<11) | ((n)<<5) | (d))  // Vd.4S += Vn.4H * Vm.H[i]
#define SMLAL2H(m, i, n, d) WORD $(0x4F402000 | ((((i)>>1)&1)<<21) | (((i)&1)<<20) | ((m)<<16) | (((i)>>2)<<11) | ((n)<<5) | (d)) // Vd.4S += Vn.8H[4:] * Vm.H[i]
#define SMULL2H(m, i, n, d) WORD $(0x4F40A000 | ((((i)>>1)&1)<<21) | (((i)&1)<<20) | ((m)<<16) | (((i)>>2)<<11) | ((n)<<5) | (d)) // Vd.4S = Vn.8H[4:] * Vm.H[i]
#define FMULS(m, i, n, d) WORD $(0x4F809000 | (((i)&1)<<21) | (((m)>>4)<<20) | (((m)&15)<<16) | (((i)>>1)<<11) | ((n)<<5) | (d)) // Vd.4S = Vn.4S * Vm.S[i]
#define FMLSS(m, i, n, d) WORD $(0x4F805000 | (((i)&1)<<21) | (((m)>>4)<<20) | (((m)&15)<<16) | (((i)>>1)<<11) | ((n)<<5) | (d)) // Vd.4S -= Vn.4S * Vm.S[i]

// Q4K_SCALES(W, DST): unpack the 6-bit scales and mins of the Q4_K block at W
// into 80 bytes at DST: the 4 scale-difference vectors D_0..D_3 (see
// gemmQ4K2x4Asm), the 8 mins (bytes), then d and dmin (f16) and 4 spare bytes.
#define Q4K_SCALES(W, DST) \
	MOVWU 4(W), R12 \
	MOVWU 8(W), R13 \
	MOVWU 12(W), R14 \
	LSRW $4, R14, R15 \
	ANDW $0x0f0f0f0f, R15, R15 \
	LSRW $6, R13, R16 \
	ANDW $0x03030303, R16, R16 \
	LSLW $4, R16, R16 \
	ORRW R16, R15, R15 \
	ANDW $0x3f3f3f3f, R13, R17 \
	ANDW $0x0f0f0f0f, R14, R19 \
	LSRW $6, R12, R16 \
	ANDW $0x03030303, R16, R16 \
	LSLW $4, R16, R16 \
	ORRW R16, R19, R19 \
	ANDW $0x3f3f3f3f, R12, R20 \
	ORR  R19<<32, R20, R20 \
	ORR  R15<<32, R17, R17 \
	MOVD R17, 64(DST) \
	MOVWU (W), R16 \
	MOVW R16, 72(DST) \
	VMOV R20, V16.D[0] \
	UXTL8B(16, 16) \
	UXTL2_8H(16, 17) \
	UXTL4H(16, 16) \
	ZERO(19) \
	EXT8(17, 16, 18) \
	EXT8(19, 17, 19) \
	SUB4S(18, 16, 16) \
	SUB4S(19, 17, 17) \
	DUPD(16, 0, 18) \
	DUPD(16, 1, 19) \
	DUPD(17, 0, 20) \
	DUPD(17, 1, 21) \
	VST1 [V18.S4, V19.S4, V20.S4, V21.S4], (DST)

// Q4K_COL(T0, T1, A0, A1): one activation column (64 permuted bytes at R10)
// against both rows' zipped weights (row 0: V22 V23 V16 V17; row 1: V18 V19
// V20 V21). T are the running int sums, A the scaled sums.
#define Q4K_COL(T0, T1, A0, A1) \
	VLD1.P 32(R10), [V28.B16, V29.B16] \
	SDOT(28, 22, T0) \
	SDOT(29, 23, T0) \
	SDOT(28, 18, T1) \
	SDOT(29, 19, T1) \
	VLD1.P 32(R10), [V28.B16, V29.B16] \
	SDOT(28, 16, T0) \
	SDOT(29, 17, T0) \
	SDOT(28, 20, T1) \
	SDOT(29, 21, T1) \
	MLA4S(26, T0, A0) \
	MLA4S(27, T1, A1)

// Q4K_CHUNK: 64 weights of each row (sub-blocks 2c, 2c+1), zipped so lanes 0
// and 2 of every vector belong to sub-block 2c and lanes 1 and 3 to 2c+1.
#define Q4K_CHUNK \
	VLD1.P 32(R7), [V22.B16, V23.B16] \
	VAND V31.B16, V22.B16, V16.B16 \
	VUSHR $4, V22.B16, V17.B16 \
	VAND V31.B16, V23.B16, V18.B16 \
	VUSHR $4, V23.B16, V19.B16 \
	ZIP1S(17, 16, 22) \
	ZIP2S(17, 16, 23) \
	ZIP1S(19, 18, 16) \
	ZIP2S(19, 18, 17) \
	VLD1.P 32(R8), [V18.B16, V19.B16] \
	VAND V31.B16, V18.B16, V20.B16 \
	VUSHR $4, V18.B16, V21.B16 \
	VAND V31.B16, V19.B16, V24.B16 \
	VUSHR $4, V19.B16, V25.B16 \
	ZIP1S(21, 20, 18) \
	ZIP2S(21, 20, 19) \
	ZIP1S(25, 24, 20) \
	ZIP2S(25, 24, 21) \
	VLD1.P 16(R11), [V26.S4] \
	VLD1.P 16(R13), [V27.S4] \
	Q4K_COL(0, 4, 8, 12) \
	Q4K_COL(1, 5, 9, 13) \
	Q4K_COL(2, 6, 10, 14) \
	Q4K_COL(3, 7, 11, 15)

// func q4kScalesAsm(w unsafe.Pointer, nb int, dst unsafe.Pointer)
// Unpacks nb Q4_K blocks' scales into 80 bytes each at dst (Q4K_SCALES): done
// once per weight row and reused for every group of 4 activation rows.
TEXT ·q4kScalesAsm(SB), NOSPLIT, $0-24
	MOVD w+0(FP), R0
	MOVD nb+8(FP), R3
	MOVD dst+16(FP), R1
	CBZ  R3, qsdone
qsloop:
	Q4K_SCALES(R0, R1)
	ADD  $144, R0
	ADD  $80, R1
	SUBS $1, R3
	BNE  qsloop
qsdone:
	RET

// func gemmQ4K2x4Asm(w0, w1, s0, s1, y unsafe.Pointer, nb int, out *float32)
// out[r*4+c] = dot(Q4_K row r, activation column c) for r < 2, c < 4.
// s0, s1 hold the rows' unpacked scales (q4kScalesAsm); y is one q8k4 group.
//
// Per block, the int sums T run across the 4 chunks; at chunk c each sum adds
// T*D_c to A, where D_c = S_c - S_(c+1) and S_c holds the scales of
// sub-blocks (2c, 2c+1) in lanes (0, 2) and (1, 3). Over the block that is
// sum_c S_c*(T_c - T_(c-1)): each chunk's dot products times their own
// scales, at one multiply per row, column and chunk.
TEXT ·gemmQ4K2x4Asm(SB), NOSPLIT, $0-56
	MOVD w0+0(FP), R0
	MOVD w1+8(FP), R1
	MOVD s0+16(FP), R11
	MOVD s1+24(FP), R13
	MOVD y+32(FP), R2
	MOVD nb+40(FP), R3
	MOVD out+48(FP), R4
	MOVW $0x0f0f0f0f, R6
	VDUP R6, V31.S4
	VEOR V30.B16, V30.B16, V30.B16
	VEOR V24.B16, V24.B16, V24.B16
	ADD  $16, R4, R9
	VST1 [V24.S4], (R9)
	CBZ  R3, g4done

g4loop:
	ZERO(0)
	ZERO(1)
	ZERO(2)
	ZERO(3)
	ZERO(4)
	ZERO(5)
	ZERO(6)
	ZERO(7)
	ZERO(8)
	ZERO(9)
	ZERO(10)
	ZERO(11)
	ZERO(12)
	ZERO(13)
	ZERO(14)
	ZERO(15)
	ADD  $16, R0, R7
	ADD  $16, R1, R8
	ADD  $16, R2, R10

	Q4K_CHUNK
	Q4K_CHUNK
	Q4K_CHUNK
	Q4K_CHUNK

	// int sums -> one vector per row, lane c = column c
	ADDP4S(9, 8, 8)
	ADDP4S(11, 10, 10)
	ADDP4S(10, 8, 8)
	ADDP4S(13, 12, 12)
	ADDP4S(15, 14, 14)
	ADDP4S(14, 12, 12)
	SCVTF4S(8, 8)
	SCVTF4S(12, 12)

	// mins . (32-value sums): bsum pairs at R10 are [k][c] int16
	VLD1 (R11), V0.D[0]
	VLD1 (R13), V0.D[1]
	UXTL8B(0, 1)
	UXTL2_16B(0, 2)
	VLD1 (R10), [V4.H8, V5.H8, V6.H8, V7.H8]
	SMULLH(1, 0, 4, 3)
	SMLAL2H(1, 1, 4, 3)
	SMLALH(1, 2, 5, 3)
	SMLAL2H(1, 3, 5, 3)
	SMLALH(1, 4, 6, 3)
	SMLAL2H(1, 5, 6, 3)
	SMLALH(1, 6, 7, 3)
	SMLAL2H(1, 7, 7, 3)
	SMULLH(2, 0, 4, 9)
	SMLAL2H(2, 1, 4, 9)
	SMLALH(2, 2, 5, 9)
	SMLAL2H(2, 3, 5, 9)
	SMLALH(2, 4, 6, 9)
	SMLAL2H(2, 5, 6, 9)
	SMLALH(2, 6, 7, 9)
	SMLAL2H(2, 7, 7, 9)
	SCVTF4S(3, 3)
	SCVTF4S(9, 9)

	// V10 = d0, dmin0, d1, dmin1; V11 = the 4 columns' Q8_K scales
	ADD  $8, R11
	ADD  $8, R13
	VLD1 (R11), V10.S[0]
	VLD1 (R13), V10.S[1]
	ADD  $8, R11
	ADD  $8, R13
	FCVTL4H(10, 10)
	VLD1 (R2), [V11.S4]
	FMULS(10, 0, 8, 8)
	FMLSS(10, 1, 3, 8)
	FMULS(10, 2, 12, 12)
	FMLSS(10, 3, 9, 12)
	VFMLA V11.S4, V8.S4, V30.S4
	VLD1 (R9), [V13.S4]
	VFMLA V11.S4, V12.S4, V13.S4
	VST1 [V13.S4], (R9)

	ADD  $144, R0
	ADD  $144, R1
	ADD  $1104, R2
	SUBS $1, R3
	BNE  g4loop

g4done:
	VST1 [V30.S4], (R4)
	RET

#define FCVTL2_8H(n, d) WORD $(0x4E217800 | ((n)<<5) | (d))                     // Vd.4S = float32(Vn.8H[4:] f16)
#define FADDP4S(m, n, d) WORD $(0x6E20D400 | ((m)<<16) | ((n)<<5) | (d))

// F16C(A0, A1, B0, B1): one activation column's next 16 floats (at R10,
// then R10 += stride) against both rows' converted weights (row 0: V0-V3,
// row 1: V4-V7); A are row 0's two accumulators, B row 1's.
#define F16C(A0, A1, B0, B1) \
	VLD1 (R10), [V24.S4, V25.S4, V26.S4, V27.S4] \
	ADD  R5, R10 \
	VFMLA V0.S4, V24.S4, A0.S4 \
	VFMLA V1.S4, V25.S4, A1.S4 \
	VFMLA V2.S4, V26.S4, A0.S4 \
	VFMLA V3.S4, V27.S4, A1.S4 \
	VFMLA V4.S4, V24.S4, B0.S4 \
	VFMLA V5.S4, V25.S4, B1.S4 \
	VFMLA V6.S4, V26.S4, B0.S4 \
	VFMLA V7.S4, V27.S4, B1.S4

// func gemmF16x2x4Asm(w0, w1, y unsafe.Pointer, stride, n int, out *float32)
// out[r*4+c] = sum_i f16(w_r[i]) * y[c*stride bytes + 4i] for r < 2, c < 4;
// n is a multiple of 16.
TEXT ·gemmF16x2x4Asm(SB), NOSPLIT, $0-48
	MOVD w0+0(FP), R0
	MOVD w1+8(FP), R1
	MOVD y+16(FP), R2
	MOVD stride+24(FP), R5
	MOVD n+32(FP), R3
	MOVD out+40(FP), R4
	LSR  $4, R3
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	CBZ  R3, f2done
f2loop:
	VLD1.P 32(R0), [V28.H8, V29.H8]
	VLD1.P 32(R1), [V30.H8, V31.H8]
	FCVTL4H(28, 0)
	FCVTL2_8H(28, 1)
	FCVTL4H(29, 2)
	FCVTL2_8H(29, 3)
	FCVTL4H(30, 4)
	FCVTL2_8H(30, 5)
	FCVTL4H(31, 6)
	FCVTL2_8H(31, 7)
	MOVD R2, R10
	F16C(V8, V9, V16, V17)
	F16C(V10, V11, V18, V19)
	F16C(V12, V13, V20, V21)
	F16C(V14, V15, V22, V23)
	ADD  $64, R2
	SUBS $1, R3
	BNE  f2loop
f2done:
	VFADD V9.S4, V8.S4, V8.S4
	VFADD V11.S4, V10.S4, V10.S4
	VFADD V13.S4, V12.S4, V12.S4
	VFADD V15.S4, V14.S4, V14.S4
	VFADD V17.S4, V16.S4, V16.S4
	VFADD V19.S4, V18.S4, V18.S4
	VFADD V21.S4, V20.S4, V20.S4
	VFADD V23.S4, V22.S4, V22.S4
	FADDP4S(10, 8, 8)
	FADDP4S(14, 12, 12)
	FADDP4S(12, 8, 8)
	FADDP4S(18, 16, 16)
	FADDP4S(22, 20, 20)
	FADDP4S(20, 16, 16)
	VST1 [V8.S4], (R4)
	ADD  $16, R4
	VST1 [V16.S4], (R4)
	RET

#define ZIP1D(m, n, d) WORD $(0x4EC03800 | ((m)<<16) | ((n)<<5) | (d))          // Vd.2D = [n0 m0]
#define ZIP2D(m, n, d) WORD $(0x4EC07800 | ((m)<<16) | ((n)<<5) | (d))          // Vd.2D = [n1 m1]
#define SLI4(n, d) WORD $(0x6F0C5400 | ((n)<<5) | (d))                          // Vd.16B = Vn<<4 | (Vd & 0x0f)
#define BIT(m, n, d) WORD $(0x6EA01C00 | ((m)<<16) | ((n)<<5) | (d))            // Vd = Vn where Vm is set, else Vd
#define SXTL8B(n, d) WORD $(0x0F08A400 | ((n)<<5) | (d))                        // Vd.8H = Vn.8B (signed)
#define SXTL2_16B(n, d) WORD $(0x4F08A400 | ((n)<<5) | (d))
#define SXTL4H(n, d) WORD $(0x0F10A400 | ((n)<<5) | (d))
#define SXTL2_8H(n, d) WORD $(0x4F10A400 | ((n)<<5) | (d))
#define FMULE0(m, n, d) WORD $(0x4F809000 | (((m)>>4)<<20) | (((m)&15)<<16) | ((n)<<5) | (d)) // Vd.4S = Vn.4S * Vm.S[0]

// Q6_COL(T, A, W0, W1, W2, W3, D): one activation column (64 bytes at R10)
// against 4 transposed weight vectors; T += dots, then A += T*D.
#define Q6_COL(T, A, W0, W1, W2, W3, D) \
	VLD1.P 32(R10), [V20.B16, V21.B16] \
	SDOT(20, W0, T) \
	SDOT(21, W1, T) \
	VLD1.P 32(R10), [V22.B16, V23.B16] \
	SDOT(22, W2, T) \
	SDOT(23, W3, T) \
	MLA4S(D, T, A)

#define Q6_COLS(W0, W1, W2, W3, D) \
	Q6_COL(0, 4, W0, W1, W2, W3, D) \
	Q6_COL(1, 5, W0, W1, W2, W3, D) \
	Q6_COL(2, 6, W0, W1, W2, W3, D) \
	Q6_COL(3, 7, W0, W1, W2, W3, D)

// Q6_T4(A, B, C, D): transpose the 32-bit lanes of 4 vectors in place, using
// V12-V15 as scratch: afterwards vector j holds lane j of A, B, C, D.
#define Q6_T4(A, B, C, D) \
	ZIP1S(B, A, 12) \
	ZIP2S(B, A, 13) \
	ZIP1S(D, C, 14) \
	ZIP2S(D, C, 15) \
	ZIP1D(14, 12, A) \
	ZIP2D(14, 12, B) \
	ZIP1D(15, 13, C) \
	ZIP2D(15, 13, D)

// Q6_HALF(DA, DB): 128 weights: group A (values 0-63, from the low nibbles,
// sub-blocks 0-3 of the half) with scale differences DA, then group B (values
// 64-127, high nibbles, sub-blocks 4-7) with DB. Weights are q-32 in int8.
#define Q6_HALF(DA, DB) \
	VLD1.P 64(R7), [V8.B16, V9.B16, V10.B16, V11.B16] \
	VLD1.P 32(R8), [V12.B16, V13.B16] \
	VUSHR $4, V8.B16, V16.B16 \
	VUSHR $4, V9.B16, V17.B16 \
	VUSHR $4, V10.B16, V18.B16 \
	VUSHR $4, V11.B16, V19.B16 \
	BIT(29, 12, 16) \
	BIT(29, 13, 17) \
	VUSHR $2, V12.B16, V14.B16 \
	VUSHR $2, V13.B16, V15.B16 \
	BIT(29, 14, 18) \
	BIT(29, 15, 19) \
	SLI4(12, 8) \
	SLI4(13, 9) \
	SLI4(14, 10) \
	SLI4(15, 11) \
	VAND V30.B16, V8.B16, V8.B16 \
	VAND V30.B16, V9.B16, V9.B16 \
	VAND V30.B16, V10.B16, V10.B16 \
	VAND V30.B16, V11.B16, V11.B16 \
	VSUB V31.B16, V8.B16, V8.B16 \
	VSUB V31.B16, V9.B16, V9.B16 \
	VSUB V31.B16, V10.B16, V10.B16 \
	VSUB V31.B16, V11.B16, V11.B16 \
	VSUB V31.B16, V16.B16, V16.B16 \
	VSUB V31.B16, V17.B16, V17.B16 \
	VSUB V31.B16, V18.B16, V18.B16 \
	VSUB V31.B16, V19.B16, V19.B16 \
	Q6_T4(8, 9, 10, 11) \
	Q6_COLS(8, 9, 10, 11, DA) \
	Q6_T4(16, 17, 18, 19) \
	Q6_COLS(16, 17, 18, 19, DB)

// func gemmQ6K1x4Asm(w, y unsafe.Pointer, nb int, out *float32)
// out[c] = dot(Q6_K row w, activation column c) for c < 4; y is one q8k6
// group. Weight vectors are transposed so lane i of every vector belongs to
// sub-block i of its 64-value group; the int sums T run across the block's 4
// groups, adding T*(S_g - S_(g+1)) at group g (see gemmQ4K2x4Asm).
TEXT ·gemmQ6K1x4Asm(SB), NOSPLIT, $0-32
	MOVD w+0(FP), R0
	MOVD y+8(FP), R2
	MOVD nb+16(FP), R3
	MOVD out+24(FP), R4
	MOVW $0x30303030, R6
	VDUP R6, V29.S4
	MOVW $0x3f3f3f3f, R6
	VDUP R6, V30.S4
	MOVW $0x20202020, R6
	VDUP R6, V31.S4
	VEOR V28.B16, V28.B16, V28.B16
	CBZ  R3, q6done

q6loop:
	ADD  $192, R0, R7
	VLD1 (R7), [V16.B16]
	SXTL8B(16, 17)
	SXTL2_16B(16, 18)
	SXTL4H(17, 24)
	SXTL2_8H(17, 25)
	SXTL4H(18, 26)
	SXTL2_8H(18, 27)
	SUB4S(25, 24, 24)
	SUB4S(26, 25, 25)
	SUB4S(27, 26, 26)
	ZERO(0)
	ZERO(1)
	ZERO(2)
	ZERO(3)
	ZERO(4)
	ZERO(5)
	ZERO(6)
	ZERO(7)
	MOVD R0, R7
	ADD  $128, R0, R8
	ADD  $16, R2, R10

	Q6_HALF(24, 25)
	Q6_HALF(26, 27)

	ADDP4S(5, 4, 4)
	ADDP4S(7, 6, 6)
	ADDP4S(6, 4, 4)
	SCVTF4S(4, 4)
	MOVHU 208(R0), R5
	FMOVS R5, F16
	FCVTHS F16, F16
	VLD1 (R2), [V17.S4]
	FMULE0(16, 17, 17)
	VFMLA V17.S4, V4.S4, V28.S4

	ADD  $210, R0
	ADD  $1040, R2
	SUBS $1, R3
	BNE  q6loop

q6done:
	VST1 [V28.S4], (R4)
	RET
