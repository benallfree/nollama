#include "textflag.h"

// The integer codec's matvec row: sum over groups g of s[g] * (sum of 64 int8 weights times int16 mantissas).
// Weights are sign-extended to 16 bits and multiplied with SMLAL into int32 lanes: a group's sum is below 2^28, so
// nothing overflows and the result is exact, the same as the Go loop.

#define SXTL8B(n, d) WORD $(0x0F08A400 | ((n)<<5) | (d))            // Vd.8H = sext(Vn.8B low)
#define SXTL2_16B(n, d) WORD $(0x4F08A400 | ((n)<<5) | (d))         // Vd.8H = sext(Vn.16B high)
#define SMLAL4H(m, n, d) WORD $(0x0E608000 | ((m)<<16) | ((n)<<5) | (d))  // Vd.4S += Vn.4H * Vm.4H
#define SMLAL2_8H(m, n, d) WORD $(0x4E608000 | ((m)<<16) | ((n)<<5) | (d)) // Vd.4S += Vn.8H[4:] * Vm.8H[4:]
#define ADDV4S(n, d) WORD $(0x4EB1B800 | ((n)<<5) | (d))            // Sd = sum of Vn.4S

// func intDotRowAsm(q *int8, x *int16, s *int16, ng int) int64
TEXT ·intDotRowAsm(SB), NOSPLIT, $0-40
	MOVD q+0(FP), R0
	MOVD x+8(FP), R1
	MOVD s+16(FP), R2
	MOVD ng+24(FP), R3
	MOVD $0, R8
	CBZ  R3, done

grp:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	MOVD $4, R4

inner:
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 32(R1), [V7.H8, V8.H8]
	SXTL8B(4, 5)
	SXTL2_16B(4, 6)
	SMLAL4H(7, 5, 0)
	SMLAL2_8H(7, 5, 1)
	SMLAL4H(8, 6, 2)
	SMLAL2_8H(8, 6, 3)
	SUBS $1, R4, R4
	BNE  inner

	VADD V1.S4, V0.S4, V0.S4
	VADD V3.S4, V2.S4, V2.S4
	VADD V2.S4, V0.S4, V0.S4
	ADDV4S(0, 0)
	VMOV V0.S[0], R5
	SXTW R5, R5
	MOVH.P 2(R2), R6
	MUL  R6, R5, R7
	ADD  R7, R8, R8
	SUBS $1, R3, R3
	BNE  grp

done:
	MOVD R8, ret+32(FP)
	RET

#define SMLAL2S(m, n, d) WORD $(0x0EA08000 | ((m)<<16) | ((n)<<5) | (d))   // Vd.2D += Vn.2S * Vm.2S (low lanes)
#define SMLAL2_4S(m, n, d) WORD $(0x4EA08000 | ((m)<<16) | ((n)<<5) | (d)) // Vd.2D += Vn.4S[2:] * Vm.4S[2:]

// func intDotQKAsm(q *int32, k *int32, n int) int64: sum of q[i]*k[i] in int64 lanes (n a multiple of 8).
TEXT ·intDotQKAsm(SB), NOSPLIT, $0-32
	MOVD q+0(FP), R0
	MOVD k+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	LSR  $3, R2, R2
	CBZ  R2, qkdone

qkloop:
	VLD1.P 32(R0), [V4.S4, V5.S4]
	VLD1.P 32(R1), [V6.S4, V7.S4]
	SMLAL2S(6, 4, 0)
	SMLAL2_4S(6, 4, 1)
	SMLAL2S(7, 5, 2)
	SMLAL2_4S(7, 5, 3)
	SUBS $1, R2, R2
	BNE  qkloop

qkdone:
	VADD V1.D2, V0.D2, V0.D2
	VADD V3.D2, V2.D2, V2.D2
	VADD V2.D2, V0.D2, V0.D2
	VMOV V0.D[0], R3
	VMOV V0.D[1], R4
	ADD  R4, R3, R3
	MOVD R3, ret+24(FP)
	RET

// func intAxpyAsm(o *int64, w int64, v *int32, n int): o[i] += w*v[i] for n a multiple of 4 (|w| < 2^31).
TEXT ·intAxpyAsm(SB), NOSPLIT, $0-32
	MOVD o+0(FP), R0
	MOVD w+8(FP), R1
	MOVD v+16(FP), R2
	MOVD n+24(FP), R3
	VDUP R1, V9.S4
	LSR  $2, R3, R3
	CBZ  R3, axdone

axloop:
	VLD1 (R0), [V0.D2, V1.D2]
	VLD1.P 16(R2), [V4.S4]
	SMLAL2S(9, 4, 0)
	SMLAL2_4S(9, 4, 1)
	VST1.P [V0.D2, V1.D2], 32(R0)
	SUBS $1, R3, R3
	BNE  axloop

axdone:
	RET
