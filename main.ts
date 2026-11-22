// A tiny stack VM: an exact interpreter for make.fun's tiny stack ISA, version 1 (the kit's
// generator/subject/ISA.md), written against the reference semantics: 32-bit wrapping integers, a
// 256-value data stack, a 64-deep call stack, 256 words of memory, input and output, ten traps with
// their own exit codes, a 1,000,000-step limit and a 65,536-byte output limit. For every instance
// <id> in the input directory it writes <id>/stdout.bin (exactly the bytes the program wrote) and
// <id>/exit.txt (its exit code, a trap's code included).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const STACK_MAX = 256;
const CALLS_MAX = 64;
const MEM_WORDS = 256;
const STEP_LIMIT = 1_000_000;
const OUT_MAX = 65_536;

// Trap exit codes.
const T_UNDERFLOW = 2;
const T_OVERFLOW = 3;
const T_DIV_ZERO = 4;
const T_BAD_OPCODE = 5;
const T_MEMORY = 6;
const T_CALL_DEPTH = 7;
const T_RET_EMPTY = 8;
const T_PC = 9;
const T_STEPS = 10;
const T_OUTPUT = 11;

// Per-opcode [immediate bytes, values popped, values pushed]; an absent byte is not an opcode.
const META: { [op: number]: [number, number, number] } = {
  0x00: [0, 0, 0], // HALT
  0x01: [1, 0, 1], // PUSH8
  0x02: [4, 0, 1], // PUSH32
  0x03: [0, 1, 0], // POP
  0x04: [0, 1, 2], // DUP
  0x05: [0, 2, 2], // SWAP
  0x06: [0, 2, 3], // OVER
  0x07: [0, 3, 3], // ROT
  0x10: [0, 2, 1], // ADD
  0x11: [0, 2, 1], // SUB
  0x12: [0, 2, 1], // MUL
  0x13: [0, 2, 1], // DIV
  0x14: [0, 2, 1], // MOD
  0x15: [0, 2, 1], // DIVU
  0x16: [0, 2, 1], // MODU
  0x17: [0, 1, 1], // NEG
  0x18: [0, 2, 1], // AND
  0x19: [0, 2, 1], // OR
  0x1a: [0, 2, 1], // XOR
  0x1b: [0, 1, 1], // NOT
  0x1c: [0, 2, 1], // SHL
  0x1d: [0, 2, 1], // SHR
  0x1e: [0, 2, 1], // SHRU
  0x20: [0, 2, 1], // EQ
  0x21: [0, 2, 1], // LT
  0x22: [0, 2, 1], // LTU
  0x30: [2, 0, 0], // JMP
  0x31: [2, 1, 0], // JZ
  0x32: [2, 1, 0], // JNZ
  0x33: [2, 0, 0], // CALL
  0x34: [0, 0, 0], // RET
  0x40: [0, 1, 1], // LOAD
  0x41: [0, 2, 0], // STORE
  0x50: [0, 1, 0], // PRINT
  0x51: [0, 1, 0], // PUTC
  0x52: [0, 0, 1], // IN
  0x53: [0, 1, 0], // EXIT
};

/** One binary arithmetic/logic/comparison step on already-popped operands; division by zero is the caller's trap. */
function binop(op: number, a: number, b: number): number {
  switch (op) {
    case 0x10: return (a + b) | 0; // ADD, wraps
    case 0x11: return (a - b) | 0; // SUB, wraps
    case 0x12: return Math.imul(a, b); // MUL, the low 32 bits of the exact product
    case 0x13: return a === -2147483648 && b === -1 ? a : (a / b) | 0; // DIV, truncated toward zero
    case 0x14: return b === -1 ? 0 : (a % b) | 0; // MOD, the sign of a
    case 0x15: return Math.floor((a >>> 0) / (b >>> 0)) | 0; // DIVU
    case 0x16: return ((a >>> 0) % (b >>> 0)) | 0; // MODU
    case 0x18: return a & b;
    case 0x19: return a | b;
    case 0x1a: return a ^ b;
    case 0x1c: return a << (b & 31); // SHL, only the low 5 bits of the count
    case 0x1d: return a >> (b & 31); // SHR, arithmetic
    case 0x1e: return (a >>> (b & 31)) | 0; // SHRU, logical
    case 0x20: return a === b ? 1 : 0; // EQ
    case 0x21: return a < b ? 1 : 0; // LT, signed
    case 0x22: return (a >>> 0) < (b >>> 0) ? 1 : 0; // LTU, unsigned
    default: return 0;
  }
}

/** Runs one program on its input bytes to its end: the bytes it wrote and the exit code it ended with. */
function runVm(P: Uint8Array, inp: Uint8Array): { out: Uint8Array; code: number } {
  const n = P.length;
  const D = new Int32Array(STACK_MAX);
  const C = new Int32Array(CALLS_MAX);
  const M = new Int32Array(MEM_WORDS);
  const out = new Uint8Array(OUT_MAX);
  let sp = 0;
  let csp = 0;
  let olen = 0;
  let ip = 0;
  let pc = 0;
  let steps = 0;
  const end = (code: number): { out: Uint8Array; code: number } => ({ out: out.slice(0, olen), code });

  for (;;) {
    // One step, in the ISA's order: step limit, pc in range, known opcode, immediate fits.
    if (steps === STEP_LIMIT) return end(T_STEPS);
    if (pc < 0 || pc >= n) return end(T_PC);
    const op = P[pc];
    const meta = META[op];
    if (meta === undefined) return end(T_BAD_OPCODE);
    const imm = meta[0];
    if (pc + imm >= n) return end(T_PC);
    steps++;
    const next = pc + 1 + imm;
    // The instruction's stack needs are checked before its own traps and effects.
    if (sp < meta[1]) return end(T_UNDERFLOW);
    if (sp - meta[1] + meta[2] > STACK_MAX) return end(T_OVERFLOW);
    pc = next;

    if (op >= 0x10 && op <= 0x22 && op !== 0x17 && op !== 0x1b) {
      const b = D[sp - 1];
      const a = D[sp - 2];
      if (b === 0 && op >= 0x13 && op <= 0x16) return end(T_DIV_ZERO);
      sp -= 2;
      D[sp++] = binop(op, a, b);
      continue;
    }

    switch (op) {
      case 0x00: // HALT
        return end(0);
      case 0x01: // PUSH8, sign-extended
        D[sp++] = (P[next - 1] << 24) >> 24;
        break;
      case 0x02: // PUSH32, little-endian
        D[sp++] = P[next - 4] | (P[next - 3] << 8) | (P[next - 2] << 16) | (P[next - 1] << 24);
        break;
      case 0x03: // POP
        sp--;
        break;
      case 0x04: // DUP
        D[sp] = D[sp - 1];
        sp++;
        break;
      case 0x05: { // SWAP
        const t = D[sp - 1];
        D[sp - 1] = D[sp - 2];
        D[sp - 2] = t;
        break;
      }
      case 0x06: // OVER
        D[sp] = D[sp - 2];
        sp++;
        break;
      case 0x07: { // ROT: ( a b c -- b c a )
        const a = D[sp - 3];
        D[sp - 3] = D[sp - 2];
        D[sp - 2] = D[sp - 1];
        D[sp - 1] = a;
        break;
      }
      case 0x17: // NEG, wraps
        D[sp - 1] = -D[sp - 1] | 0;
        break;
      case 0x1b: // NOT
        D[sp - 1] = ~D[sp - 1];
        break;
      case 0x30: // JMP
      case 0x31: // JZ
      case 0x32: { // JNZ
        const rel = ((P[next - 2] | (P[next - 1] << 8)) << 16) >> 16;
        let take = true;
        if (op !== 0x30) {
          const v = D[--sp];
          take = op === 0x31 ? v === 0 : v !== 0;
        }
        if (take) pc = next + rel; // the target is checked at the next step, not here
        break;
      }
      case 0x33: // CALL, the immediate is an unsigned absolute address
        if (csp === CALLS_MAX) return end(T_CALL_DEPTH);
        C[csp++] = next;
        pc = P[next - 2] | (P[next - 1] << 8);
        break;
      case 0x34: // RET
        if (csp === 0) return end(T_RET_EMPTY);
        pc = C[--csp];
        break;
      case 0x40: { // LOAD
        const addr = D[sp - 1];
        if (addr < 0 || addr >= MEM_WORDS) return end(T_MEMORY);
        D[sp - 1] = M[addr];
        break;
      }
      case 0x41: { // STORE: ( v addr -- )
        const addr = D[sp - 1];
        if (addr < 0 || addr >= MEM_WORDS) return end(T_MEMORY);
        const v = D[sp - 2];
        sp -= 2;
        M[addr] = v;
        break;
      }
      case 0x50: { // PRINT, decimal ASCII; a write past the output limit writes nothing and traps
        const s = String(D[sp - 1]);
        if (olen + s.length > OUT_MAX) return end(T_OUTPUT);
        sp--;
        for (let k = 0; k < s.length; k++) out[olen++] = s.charCodeAt(k);
        break;
      }
      case 0x51: // PUTC, one byte
        if (olen + 1 > OUT_MAX) return end(T_OUTPUT);
        out[olen++] = D[--sp] & 255;
        break;
      case 0x52: // IN, -1 once the input is used up
        D[sp++] = ip < inp.length ? inp[ip++] : -1;
        break;
      case 0x53: // EXIT, the low byte of the value
        return end(D[--sp] & 255);
    }
  }
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const r = runVm(new Uint8Array(readFileSync(join(I, id, 'program.bin'))), new Uint8Array(readFileSync(join(I, id, 'input.bin'))));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), r.out);
  writeFileSync(join(O, id, 'exit.txt'), `${r.code}\n`);
}
