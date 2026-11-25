//go:build amd64

#include "textflag.h"

// AVX2 elementwise kernels for the softmax and SwiGLU steps (kernels.go has the Go versions and the contract). Each
// element goes through the same float32 operations, in the same order and with the same rounding, as expf, swiglu
// and ftz32 in Go: no FMA contraction, VROUNDPS for the floor, VDIVPS for the division. Results are bit-identical.
// Lengths are multiples of 8; the Go wrappers do the tails.

DATA ewc<>+0(SB)/4, $0xc2ae0000
DATA ewc<>+4(SB)/4, $0xc2ae0000
DATA ewc<>+8(SB)/4, $0xc2ae0000
DATA ewc<>+12(SB)/4, $0xc2ae0000
DATA ewc<>+16(SB)/4, $0xc2ae0000
DATA ewc<>+20(SB)/4, $0xc2ae0000
DATA ewc<>+24(SB)/4, $0xc2ae0000
DATA ewc<>+28(SB)/4, $0xc2ae0000
DATA ewc<>+32(SB)/4, $0x42b00000
DATA ewc<>+36(SB)/4, $0x42b00000
DATA ewc<>+40(SB)/4, $0x42b00000
DATA ewc<>+44(SB)/4, $0x42b00000
DATA ewc<>+48(SB)/4, $0x42b00000
DATA ewc<>+52(SB)/4, $0x42b00000
DATA ewc<>+56(SB)/4, $0x42b00000
DATA ewc<>+60(SB)/4, $0x42b00000
DATA ewc<>+64(SB)/4, $0x3fb8aa3b
DATA ewc<>+68(SB)/4, $0x3fb8aa3b
DATA ewc<>+72(SB)/4, $0x3fb8aa3b
DATA ewc<>+76(SB)/4, $0x3fb8aa3b
DATA ewc<>+80(SB)/4, $0x3fb8aa3b
DATA ewc<>+84(SB)/4, $0x3fb8aa3b
DATA ewc<>+88(SB)/4, $0x3fb8aa3b
DATA ewc<>+92(SB)/4, $0x3fb8aa3b
DATA ewc<>+96(SB)/4, $0x3f000000
DATA ewc<>+100(SB)/4, $0x3f000000
DATA ewc<>+104(SB)/4, $0x3f000000
DATA ewc<>+108(SB)/4, $0x3f000000
DATA ewc<>+112(SB)/4, $0x3f000000
DATA ewc<>+116(SB)/4, $0x3f000000
DATA ewc<>+120(SB)/4, $0x3f000000
DATA ewc<>+124(SB)/4, $0x3f000000
DATA ewc<>+128(SB)/4, $0x3f318000
DATA ewc<>+132(SB)/4, $0x3f318000
DATA ewc<>+136(SB)/4, $0x3f318000
DATA ewc<>+140(SB)/4, $0x3f318000
DATA ewc<>+144(SB)/4, $0x3f318000
DATA ewc<>+148(SB)/4, $0x3f318000
DATA ewc<>+152(SB)/4, $0x3f318000
DATA ewc<>+156(SB)/4, $0x3f318000
DATA ewc<>+160(SB)/4, $0xb95e8083
DATA ewc<>+164(SB)/4, $0xb95e8083
DATA ewc<>+168(SB)/4, $0xb95e8083
DATA ewc<>+172(SB)/4, $0xb95e8083
DATA ewc<>+176(SB)/4, $0xb95e8083
DATA ewc<>+180(SB)/4, $0xb95e8083
DATA ewc<>+184(SB)/4, $0xb95e8083
DATA ewc<>+188(SB)/4, $0xb95e8083
DATA ewc<>+192(SB)/4, $0x39506967
DATA ewc<>+196(SB)/4, $0x39506967
DATA ewc<>+200(SB)/4, $0x39506967
DATA ewc<>+204(SB)/4, $0x39506967
DATA ewc<>+208(SB)/4, $0x39506967
DATA ewc<>+212(SB)/4, $0x39506967
DATA ewc<>+216(SB)/4, $0x39506967
DATA ewc<>+220(SB)/4, $0x39506967
DATA ewc<>+224(SB)/4, $0x3ab743ce
DATA ewc<>+228(SB)/4, $0x3ab743ce
DATA ewc<>+232(SB)/4, $0x3ab743ce
DATA ewc<>+236(SB)/4, $0x3ab743ce
DATA ewc<>+240(SB)/4, $0x3ab743ce
DATA ewc<>+244(SB)/4, $0x3ab743ce
DATA ewc<>+248(SB)/4, $0x3ab743ce
DATA ewc<>+252(SB)/4, $0x3ab743ce
DATA ewc<>+256(SB)/4, $0x3c088908
DATA ewc<>+260(SB)/4, $0x3c088908
DATA ewc<>+264(SB)/4, $0x3c088908
DATA ewc<>+268(SB)/4, $0x3c088908
DATA ewc<>+272(SB)/4, $0x3c088908
DATA ewc<>+276(SB)/4, $0x3c088908
DATA ewc<>+280(SB)/4, $0x3c088908
DATA ewc<>+284(SB)/4, $0x3c088908
DATA ewc<>+288(SB)/4, $0x3d2aa9c1
DATA ewc<>+292(SB)/4, $0x3d2aa9c1
DATA ewc<>+296(SB)/4, $0x3d2aa9c1
DATA ewc<>+300(SB)/4, $0x3d2aa9c1
DATA ewc<>+304(SB)/4, $0x3d2aa9c1
DATA ewc<>+308(SB)/4, $0x3d2aa9c1
DATA ewc<>+312(SB)/4, $0x3d2aa9c1
DATA ewc<>+316(SB)/4, $0x3d2aa9c1
DATA ewc<>+320(SB)/4, $0x3e2aaaaa
DATA ewc<>+324(SB)/4, $0x3e2aaaaa
DATA ewc<>+328(SB)/4, $0x3e2aaaaa
DATA ewc<>+332(SB)/4, $0x3e2aaaaa
DATA ewc<>+336(SB)/4, $0x3e2aaaaa
DATA ewc<>+340(SB)/4, $0x3e2aaaaa
DATA ewc<>+344(SB)/4, $0x3e2aaaaa
DATA ewc<>+348(SB)/4, $0x3e2aaaaa
DATA ewc<>+352(SB)/4, $0x3f000000
DATA ewc<>+356(SB)/4, $0x3f000000
DATA ewc<>+360(SB)/4, $0x3f000000
DATA ewc<>+364(SB)/4, $0x3f000000
DATA ewc<>+368(SB)/4, $0x3f000000
DATA ewc<>+372(SB)/4, $0x3f000000
DATA ewc<>+376(SB)/4, $0x3f000000
DATA ewc<>+380(SB)/4, $0x3f000000
DATA ewc<>+384(SB)/4, $0x3f800000
DATA ewc<>+388(SB)/4, $0x3f800000
DATA ewc<>+392(SB)/4, $0x3f800000
DATA ewc<>+396(SB)/4, $0x3f800000
DATA ewc<>+400(SB)/4, $0x3f800000
DATA ewc<>+404(SB)/4, $0x3f800000
DATA ewc<>+408(SB)/4, $0x3f800000
DATA ewc<>+412(SB)/4, $0x3f800000
DATA ewc<>+416(SB)/4, $0x0000007f
DATA ewc<>+420(SB)/4, $0x0000007f
DATA ewc<>+424(SB)/4, $0x0000007f
DATA ewc<>+428(SB)/4, $0x0000007f
DATA ewc<>+432(SB)/4, $0x0000007f
DATA ewc<>+436(SB)/4, $0x0000007f
DATA ewc<>+440(SB)/4, $0x0000007f
DATA ewc<>+444(SB)/4, $0x0000007f
DATA ewc<>+448(SB)/4, $0x00800000
DATA ewc<>+452(SB)/4, $0x00800000
DATA ewc<>+456(SB)/4, $0x00800000
DATA ewc<>+460(SB)/4, $0x00800000
DATA ewc<>+464(SB)/4, $0x00800000
DATA ewc<>+468(SB)/4, $0x00800000
DATA ewc<>+472(SB)/4, $0x00800000
DATA ewc<>+476(SB)/4, $0x00800000
DATA ewc<>+480(SB)/4, $0x7fffffff
DATA ewc<>+484(SB)/4, $0x7fffffff
DATA ewc<>+488(SB)/4, $0x7fffffff
DATA ewc<>+492(SB)/4, $0x7fffffff
DATA ewc<>+496(SB)/4, $0x7fffffff
DATA ewc<>+500(SB)/4, $0x7fffffff
DATA ewc<>+504(SB)/4, $0x7fffffff
DATA ewc<>+508(SB)/4, $0x7fffffff
DATA ewc<>+512(SB)/4, $0x80000000
DATA ewc<>+516(SB)/4, $0x80000000
DATA ewc<>+520(SB)/4, $0x80000000
DATA ewc<>+524(SB)/4, $0x80000000
DATA ewc<>+528(SB)/4, $0x80000000
DATA ewc<>+532(SB)/4, $0x80000000
DATA ewc<>+536(SB)/4, $0x80000000
DATA ewc<>+540(SB)/4, $0x80000000
GLOBL ewc<>(SB), RODATA|NOPTR, $544

#define M87 ewc<>+0(SB)
#define C88 ewc<>+32(SB)
#define LOG2E ewc<>+64(SB)
#define HALF ewc<>+96(SB)
#define LN2HI ewc<>+128(SB)
#define LN2LO ewc<>+160(SB)
#define P0 ewc<>+192(SB)
#define P1 ewc<>+224(SB)
#define P2 ewc<>+256(SB)
#define P3 ewc<>+288(SB)
#define P4 ewc<>+320(SB)
#define P5 ewc<>+352(SB)
#define ONE ewc<>+384(SB)
#define BIAS ewc<>+416(SB)
#define FLTMIN ewc<>+448(SB)
#define ABS ewc<>+480(SB)
#define SIGN ewc<>+512(SB)

DATA ew16<>+0(SB)/8, $0x0001000100010001
DATA ew16<>+8(SB)/8, $0x0001000100010001
DATA ew16<>+16(SB)/8, $0x0001000100010001
DATA ew16<>+24(SB)/8, $0x0001000100010001
GLOBL ew16<>(SB), RODATA|NOPTR, $32

// EXPF sets Y0 = expf(Y0), using Y1-Y3 and Y5; Y15 holds C88.
#define EXPF \
	VCMPPS $1, M87, Y0, Y5; \
	VMINPS Y0, Y15, Y0; \
	VMULPS LOG2E, Y0, Y1; \
	VADDPS HALF, Y1, Y1; \
	VROUNDPS $1, Y1, Y1; \
	VMULPS LN2HI, Y1, Y2; \
	VSUBPS Y2, Y0, Y2; \
	VMULPS LN2LO, Y1, Y3; \
	VSUBPS Y3, Y2, Y2; \
	VMULPS P0, Y2, Y3; \
	VADDPS P1, Y3, Y3; \
	VMULPS Y2, Y3, Y3; \
	VADDPS P2, Y3, Y3; \
	VMULPS Y2, Y3, Y3; \
	VADDPS P3, Y3, Y3; \
	VMULPS Y2, Y3, Y3; \
	VADDPS P4, Y3, Y3; \
	VMULPS Y2, Y3, Y3; \
	VADDPS P5, Y3, Y3; \
	VMULPS Y2, Y3, Y3; \
	VMULPS Y2, Y3, Y3; \
	VADDPS Y2, Y3, Y3; \
	VADDPS ONE, Y3, Y3; \
	VCVTTPS2DQ Y1, Y1; \
	VPADDD BIAS, Y1, Y1; \
	VPSLLD $23, Y1, Y1; \
	VMULPS Y1, Y3, Y0; \
	VANDNPS Y0, Y5, Y0

// FTZ(r, t) flushes subnormal lanes of r to signed zero, as ftz32.
#define FTZ(r, t) \
	VANDPS ABS, r, t; \
	VCMPPS $1, FLTMIN, t, t; \
	VANDPS ABS, t, t; \
	VANDNPS r, t, r

// func expSubAVX2(x *float32, n int, m float32)
// x[j] = expf(x[j] - m) for j < n.
TEXT ·expSubAVX2(SB), NOSPLIT, $0-20
	MOVQ x+0(FP), DI
	MOVQ n+8(FP), CX
	VBROADCASTSS m+16(FP), Y14
	VMOVUPS C88, Y15
	SHRQ $3, CX
	JZ esdone
esloop:
	VMOVUPS (DI), Y0
	VSUBPS Y14, Y0, Y0
	EXPF
	VMOVUPS Y0, (DI)
	ADDQ $32, DI
	DECQ CX
	JNZ esloop
esdone:
	VZEROUPPER
	RET

// func swigluAVX2(dst, gate, up *float32, n int)
// dst[i] = swiglu(gate[i], up[i]) for i < n.
TEXT ·swigluAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ gate+8(FP), SI
	MOVQ up+16(FP), DX
	MOVQ n+24(FP), CX
	VMOVUPS C88, Y15
	SHRQ $3, CX
	JZ swdone
swloop:
	VMOVUPS (SI), Y6
	FTZ(Y6, Y7)              // g = ftz32(g)
	VXORPS SIGN, Y6, Y0      // -g
	EXPF
	VADDPS ONE, Y0, Y0       // 1 + expf(-g)
	VDIVPS Y0, Y6, Y6        // g / (1 + expf(-g))
	FTZ(Y6, Y7)
	VMOVUPS (DX), Y8
	FTZ(Y8, Y7)
	VMULPS Y8, Y6, Y6
	FTZ(Y6, Y7)
	VMOVUPS Y6, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	DECQ CX
	JNZ swloop
swdone:
	VZEROUPPER
	RET

// func scaleMaxAVX2(x *float32, n int, s float32) float32
// x[j] *= s for j < n (n > 0); returns the largest x[j].
TEXT ·scaleMaxAVX2(SB), NOSPLIT, $0-28
	MOVQ x+0(FP), DI
	MOVQ n+8(FP), CX
	VBROADCASTSS s+16(FP), Y14
	VMULPS (DI), Y14, Y1
	VMOVUPS Y1, (DI)
	ADDQ $32, DI
	SHRQ $3, CX
	DECQ CX
	JZ smdone
smloop:
	VMULPS (DI), Y14, Y0
	VMOVUPS Y0, (DI)
	VMAXPS Y0, Y1, Y1
	ADDQ $32, DI
	DECQ CX
	JNZ smloop
smdone:
	VEXTRACTF128 $1, Y1, X0
	VMAXPS X0, X1, X1
	VPERMILPS $0x4e, X1, X0
	VMAXPS X0, X1, X1
	VPERMILPS $0xb1, X1, X0
	VMAXPS X0, X1, X1
	VMOVSS X1, ret+24(FP)
	VZEROUPPER
	RET

// func mulFTZAVX2(x *float32, n int, s float32)
// x[j] = ftz32(x[j] * s) for j < n.
TEXT ·mulFTZAVX2(SB), NOSPLIT, $0-20
	MOVQ x+0(FP), DI
	MOVQ n+8(FP), CX
	VBROADCASTSS s+16(FP), Y14
	SHRQ $3, CX
	JZ mfdone
mfloop:
	VMULPS (DI), Y14, Y0
	FTZ(Y0, Y1)
	VMOVUPS Y0, (DI)
	ADDQ $32, DI
	DECQ CX
	JNZ mfloop
mfdone:
	VZEROUPPER
	RET

// func rmsSumSqAVX2(x *float32, n int, acc *[32]float32)
// acc[i%32] = ftz32(fma32(v, v, acc[i%32])) with v = ftz32(x[i]), i in order; n a multiple of 32.
TEXT ·rmsSumSqAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	MOVQ acc+16(FP), DI
	VMOVUPS (DI), Y8
	VMOVUPS 32(DI), Y9
	VMOVUPS 64(DI), Y10
	VMOVUPS 96(DI), Y11
	SHRQ $5, CX
	JZ rsdone
rsloop:
	VMOVUPS (SI), Y0
	FTZ(Y0, Y4)
	VFMADD231PS Y0, Y0, Y8
	FTZ(Y8, Y4)
	VMOVUPS 32(SI), Y1
	FTZ(Y1, Y5)
	VFMADD231PS Y1, Y1, Y9
	FTZ(Y9, Y5)
	VMOVUPS 64(SI), Y2
	FTZ(Y2, Y6)
	VFMADD231PS Y2, Y2, Y10
	FTZ(Y10, Y6)
	VMOVUPS 96(SI), Y3
	FTZ(Y3, Y7)
	VFMADD231PS Y3, Y3, Y11
	FTZ(Y11, Y7)
	ADDQ $128, SI
	DECQ CX
	JNZ rsloop
rsdone:
	VMOVUPS Y8, (DI)
	VMOVUPS Y9, 32(DI)
	VMOVUPS Y10, 64(DI)
	VMOVUPS Y11, 96(DI)
	VZEROUPPER
	RET

// func rmsScaleAVX2(out, x, w *float32, n int, s float32)
// out[i] = ftz32(ftz32(ftz32(x[i])*s) * ftz32(w[i])); n a multiple of 8.
TEXT ·rmsScaleAVX2(SB), NOSPLIT, $0-36
	MOVQ out+0(FP), DI
	MOVQ x+8(FP), SI
	MOVQ w+16(FP), DX
	MOVQ n+24(FP), CX
	VBROADCASTSS s+32(FP), Y14
	SHRQ $3, CX
	JZ rcdone
rcloop:
	VMOVUPS (SI), Y0
	FTZ(Y0, Y2)
	VMULPS Y14, Y0, Y0
	FTZ(Y0, Y2)
	VMOVUPS (DX), Y1
	FTZ(Y1, Y2)
	VMULPS Y1, Y0, Y0
	FTZ(Y0, Y2)
	VMOVUPS Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	DECQ CX
	JNZ rcloop
rcdone:
	VZEROUPPER
	RET

// func q8kAbsMaxAVX2(x *float32) (amax, max float32)
// Over 256 values: amax = the largest |x[i]| (0 if none is above 0), max = the first x[i] with |x[i]| == amax.
TEXT ·q8kAbsMaxAVX2(SB), NOSPLIT, $0-16
	MOVQ x+0(FP), SI
	VMOVUPS ABS, Y15
	VXORPS Y1, Y1, Y1
	MOVQ $0, AX
amloop:
	VANDPS (SI)(AX*1), Y15, Y0
	VMAXPS Y0, Y1, Y1
	ADDQ $32, AX
	CMPQ AX, $1024
	JLT amloop
	VEXTRACTF128 $1, Y1, X0
	VMAXPS X0, X1, X1
	VPERMILPS $0x4e, X1, X0
	VMAXPS X0, X1, X1
	VPERMILPS $0xb1, X1, X0
	VMAXPS X0, X1, X1
	VMOVSS X1, amax+8(FP)
	VBROADCASTSS X1, Y1
	MOVQ $0, AX
	VMOVSS (SI), X2
amscan:
	VANDPS (SI)(AX*1), Y15, Y0
	VCMPPS $0, Y1, Y0, Y0
	VMOVMSKPS Y0, BX
	TESTL BX, BX
	JNZ amfound
	ADDQ $32, AX
	CMPQ AX, $1024
	JLT amscan
	JMP amout
amfound:
	BSFL BX, BX
	LEAQ (AX)(BX*4), AX
	VMOVSS (SI)(AX*1), X2
amout:
	VMOVSS X2, max+12(FP)
	VZEROUPPER
	RET

// func q8kQuantAVX2(x *float32, iscale float32, ob *byte)
// Writes qs (int8(clamp(roundEven(iscale*x[j])))) and the 16-value bsums of one block_q8_K; d is left to Go.
TEXT ·q8kQuantAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	VBROADCASTSS iscale+8(FP), Y14
	MOVQ ob+16(FP), DI
	VMOVDQU ew16<>(SB), Y13
	MOVQ $0, CX
qqloop:
	VMULPS (SI), Y14, Y0
	VMULPS 32(SI), Y14, Y1
	VCVTPS2DQ Y0, Y0
	VCVTPS2DQ Y1, Y1
	VPACKSSDW Y1, Y0, Y2
	VPERMQ $0xd8, Y2, Y2
	VEXTRACTI128 $1, Y2, X3
	VPACKSSWB X3, X2, X4
	VMOVDQU X4, 4(DI)(CX*1)
	VPMOVSXBW X4, Y5
	VPMADDWD Y13, Y5, Y5
	VEXTRACTI128 $1, Y5, X6
	VPADDD X6, X5, X5
	VPHADDD X5, X5, X5
	VPHADDD X5, X5, X5
	VMOVD X5, AX
	MOVQ CX, BX
	SHRQ $3, BX
	MOVW AX, 260(DI)(BX*1)
	ADDQ $64, SI
	ADDQ $16, CX
	CMPQ CX, $256
	JLT qqloop
	VZEROUPPER
	RET
