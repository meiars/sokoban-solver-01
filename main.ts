// Sokoban solver for make.fun's sokoban-solver-01-run-20261121.
// Push-optimal per room, stitched by hallway walks.
//
// The level is a one-square-wide hallway (one full row) with up to four
// rooms off it (6x6 slots, one door each, 3-4 boxes and as many goals per
// room). A box pushed out of its door into the hallway can never be pushed
// back, so in an optimal solution every box stays in its room and the
// level's push-optimal count is the sum of its rooms' optima. This solver
// therefore splits the level into rooms (connected floor components once
// the hallway row is removed), solves each room independently with
// push-based A* (admissible min-cost box-goal assignment heuristic,
// dead-square + 2x2-block pruning), and stitches the rooms with BFS
// hallway walks. No clock, no randomness, no network: every budget is a
// count of expanded states, so two runs produce identical bytes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Push budgets are counts of expanded states, never wall time.
const ROOM_BUDGET = 400_000;
const GREEDY_BUDGET = 200_000;
const GLOBAL_BUDGET = 2_000_000;
let globalUsed = 0;

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch !== '#') wall[i] = 0;
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/** Flood fill from `from` over cells where `open[i]` is set. Returns the cells, sorted. */
function flood(w: number, open: Uint8Array, from: number): number[] {
  const seen = new Uint8Array(open.length);
  const out: number[] = [from];
  seen[from] = 1;
  for (let i = 0; i < out.length; i++) {
    const c = out[i];
    const x = c % w;
    const y = (c / w) | 0;
    if (y > 0 && open[c - w] && !seen[c - w]) { seen[c - w] = 1; out.push(c - w); }
    if (y < (open.length / w | 0) - 1 && open[c + w] && !seen[c + w]) { seen[c + w] = 1; out.push(c + w); }
    if (x > 0 && open[c - 1] && !seen[c - 1]) { seen[c - 1] = 1; out.push(c - 1); }
    if (x < w - 1 && open[c + 1] && !seen[c + 1]) { seen[c + 1] = 1; out.push(c + 1); }
  }
  out.sort((a, b) => a - b);
  return out;
}

/** Shortest walk (LURD string) from `from` to `to` over `open` cells, or null. */
function walkPath(w: number, h: number, open: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(open.length).fill(-1);
  const prevCh = new Array<string>(open.length).fill('');
  const q: number[] = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const c = q[i];
    for (const [dx, dy, ch] of DIRS) {
      const nx = (c % w) + dx;
      const ny = ((c / w) | 0) + dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (!open[n] || prev[n] !== -1) continue;
      prev[n] = c;
      prevCh[n] = ch;
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          s = prevCh[cur] + s;
          cur = prev[cur];
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

/** BFS distances from `src` over cells where `open[i]` is set. -1 = unreachable. */
function bfsDist(w: number, open: Uint8Array, src: number): Int32Array {
  const dist = new Int32Array(open.length).fill(-1);
  const q: number[] = [src];
  dist[src] = 0;
  for (let i = 0; i < q.length; i++) {
    const c = q[i];
    const x = c % w;
    if (c >= w && open[c - w] && dist[c - w] < 0) { dist[c - w] = dist[c] + 1; q.push(c - w); }
    if (c + w < open.length && open[c + w] && dist[c + w] < 0) { dist[c + w] = dist[c] + 1; q.push(c + w); }
    if (x > 0 && open[c - 1] && dist[c - 1] < 0) { dist[c - 1] = dist[c] + 1; q.push(c - 1); }
    if (x < w - 1 && open[c + 1] && dist[c + 1] < 0) { dist[c + 1] = dist[c] + 1; q.push(c + 1); }
  }
  return dist;
}

/** All permutations of [0..n). */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = Array.from({ length: n }, (_, i) => i);
  const gen = (k: number): void => {
    if (k === n) { out.push([...a]); return; }
    for (let i = k; i < n; i++) {
      [a[k], a[i]] = [a[i], a[k]];
      gen(k + 1);
      [a[k], a[i]] = [a[i], a[k]];
    }
  };
  gen(0);
  return out;
}

/** A binary heap on [f, h, seq]: smallest f, then smallest h, then earliest. */
class Heap {
  private a: [number, number, number, number][] = [];
  get size(): number { return this.a.length; }
  push(e: [number, number, number, number]): void {
    const a = this.a;
    a.push(e);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (cmp(a[i], a[p]) >= 0) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop(): [number, number, number, number] {
    const a = this.a;
    const top = a[0];
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && cmp(a[l], a[m]) < 0) m = l;
        if (r < a.length && cmp(a[r], a[m]) < 0) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
}
function cmp(x: [number, number, number, number], y: [number, number, number, number]): number {
  return x[0] - y[0] || x[1] - y[1] || x[2] - y[2];
}

interface RoomCtx {
  w: number;
  h: number;
  /** Cells a box may occupy (the room component). */
  boxCells: Uint8Array;
  /** Cells the player may walk (room + hallway). */
  roam: Uint8Array;
  goals: number[];
  /** dist[g][cell]: BFS distance from goal g over boxCells. */
  dist: Int32Array[];
  /** Cells a box can never reach a goal from (reverse-push reachability). */
  dead: Uint8Array;
  perms: number[][];
}

function makeRoomCtx(
  lv: Level,
  boxCells: Uint8Array,
  roam: Uint8Array,
  goals: number[],
): RoomCtx {
  const { w } = lv;
  const dist = goals.map((g) => bfsDist(w, boxCells, g));
  // Dead squares: reverse BFS from the goals over the box graph. A box at
  // p can move to p+d when p+d is a box cell and p-d is a cell the player
  // can stand on, so a goal's predecessors are p = g-d with p a box cell
  // and g-2d a roam cell. Cells this never reaches can never reach a goal.
  const live = new Uint8Array(boxCells.length);
  const q: number[] = [];
  for (const g of goals) {
    if (!live[g]) { live[g] = 1; q.push(g); }
  }
  for (let i = 0; i < q.length; i++) {
    const x = q[i];
    const xx = x % w;
    const xy = (x / w) | 0;
    for (const [dx, dy] of DIRS) {
      const px = xx - dx;
      const py = xy - dy;
      if (px < 0 || px >= w || py < 0 || py >= lv.h) continue;
      const p = py * w + px;
      const sx = xx - 2 * dx;
      const sy = xy - 2 * dy;
      if (sx < 0 || sx >= w || sy < 0 || sy >= lv.h) continue;
      if (!boxCells[p] || !roam[sy * w + sx] || live[p]) continue;
      live[p] = 1;
      q.push(p);
    }
  }
  const dead = new Uint8Array(boxCells.length);
  for (let i = 0; i < boxCells.length; i++) if (boxCells[i] && !live[i]) dead[i] = 1;
  return { w, h: lv.h, boxCells, roam, goals, dist, dead, perms: permutations(goals.length) };
}

/** Min-cost assignment of boxes to goals; Infinity when no assignment exists. */
function heuristic(ctx: RoomCtx, boxes: number[]): number {
  const n = boxes.length;
  let best = Infinity;
  for (const perm of ctx.perms) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const d = ctx.dist[perm[i]][boxes[i]];
      if (d < 0) { s = Infinity; break; }
      s += d;
    }
    if (s < best) best = s;
  }
  return best;
}

/** True when the moved box completes a 2x2 wall/box block with a box off-goal in it. */
function blocks2x2(ctx: RoomCtx, boxes: Set<number>, moved: number, goalSet: Set<number>): boolean {
  const { w, h } = ctx;
  const occ = (x: number, y: number): boolean => {
    if (x < 0 || x >= w || y < 0 || y >= h) return true; // outside the board is wall
    const i = y * w + x;
    return ctx.boxCells[i] === 0 || boxes.has(i);
  };
  const mx = moved % w;
  const my = (moved / w) | 0;
  for (const [ox, oy] of [[-1, -1], [0, -1], [-1, 0], [0, 0]] as const) {
    const x0 = mx + ox;
    const y0 = my + oy;
    if (!occ(x0, y0) || !occ(x0 + 1, y0) || !occ(x0, y0 + 1) || !occ(x0 + 1, y0 + 1)) continue;
    for (const [cx, cy] of [[x0, y0], [x0 + 1, y0], [x0, y0 + 1], [x0 + 1, y0 + 1]] as const) {
      const c = cy * w + cx;
      if (boxes.has(c) && !goalSet.has(c)) return true;
    }
  }
  return false;
}

interface SearchResult {
  moves: string;
  pushes: number;
}

/**
 * Push A* over one room (or the whole level, when called with the full
 * floor as boxCells). `startPlayer` is the player's cell when the search
 * begins. Returns the move string, or null when the budget runs out.
 * `greedy` switches the priority to h-only (best-first) for a fallback.
 */
function solveRoom(
  ctx: RoomCtx,
  boxes0: number[],
  startPlayer: number,
  budget: number,
  greedy: boolean,
): SearchResult | null {
  const goalSet = new Set(ctx.goals);
  const n = boxes0.length;
  if (n === 0) return { moves: '', pushes: 0 };
  if (boxes0.some((b) => ctx.dead[b])) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => goalSet.has(b));
  if (solved(boxes0)) return { moves: '', pushes: 0 };

  // State arrays, indexed by state id.
  const sBoxes: number[][] = [];
  const sPlayer: number[] = [];
  const sG: number[] = [];
  const sParent: number[] = [];
  const sStep: string[] = [];
  const best = new Map<string, number>();
  const heap = new Heap();
  let seq = 0;

  const keyOf = (bs: number[], player: number, boxSet: Set<number>): string => {
    // Canonical player cell: the smallest cell of the player's reachable
    // region (roam minus boxes). Two states that differ only by where the
    // player stands inside one region are the same state.
    const open = new Uint8Array(ctx.roam.length);
    for (let i = 0; i < ctx.roam.length; i++) open[i] = ctx.roam[i] && !boxSet.has(i) ? 1 : 0;
    if (!open[player]) return ''; // player stands on a box: unreachable state
    const region = flood(ctx.w, open, player);
    return bs.join(',') + '|' + region[0];
  };

  const h0 = heuristic(ctx, boxes0);
  if (!Number.isFinite(h0)) return null;
  const k0 = keyOf(boxes0, startPlayer, new Set(boxes0));
  if (k0 === '') return null;
  sBoxes.push([...boxes0]);
  sPlayer.push(startPlayer);
  sG.push(0);
  sParent.push(-1);
  sStep.push('');
  best.set(k0, 0);
  heap.push([greedy ? h0 : h0, h0, seq++, 0]);

  let expanded = 0;
  while (heap.size > 0) {
    if (expanded >= budget || globalUsed >= GLOBAL_BUDGET) return null;
    const [, , , id] = heap.pop();
    const bs = sBoxes[id];
    const player = sPlayer[id];
    const g = sG[id];
    // The heap may hold a stale entry superseded by a shorter path.
    const boxSet = new Set(bs);
    const k = keyOf(bs, player, boxSet);
    if (best.get(k) !== g) continue;
    expanded++;
    globalUsed++;
    if (solved(bs)) {
      // Reconstruct the move string from the parent chain.
      const parts: string[] = [];
      let cur = id;
      while (cur > 0) {
        parts.push(sStep[cur]);
        cur = sParent[cur];
      }
      parts.reverse();
      return { moves: parts.join(''), pushes: g };
    }
    // Player's reachable region and the open cells for walks.
    const open = new Uint8Array(ctx.roam.length);
    for (let i = 0; i < ctx.roam.length; i++) open[i] = ctx.roam[i] && !boxSet.has(i) ? 1 : 0;
    if (!open[player]) continue;
    const region = new Set(flood(ctx.w, open, player));
    for (let bi = 0; bi < n; bi++) {
      const b = bs[bi];
      const bx = b % ctx.w;
      const by = (b / ctx.w) | 0;
      for (const [dx, dy, ch] of DIRS) {
        const dest = (by + dy) * ctx.w + (bx + dx);
        const stance = (by - dy) * ctx.w + (bx - dx);
        if (bx + dx < 0 || bx + dx >= ctx.w || by + dy < 0 || by + dy >= ctx.h) continue;
        if (!ctx.boxCells[dest] || boxSet.has(dest) || ctx.dead[dest]) continue;
        if (bx - dx < 0 || bx - dx >= ctx.w || by - dy < 0 || by - dy >= ctx.h) continue;
        if (!region.has(stance)) continue;
        const walk = walkPath(ctx.w, ctx.h, open, player, stance);
        if (walk === null) continue;
        const nbs = [...bs];
        nbs[bi] = dest;
        nbs.sort((a, b2) => a - b2);
        const nSet = new Set(nbs);
        if (blocks2x2(ctx, nSet, dest, goalSet)) continue;
        const h = heuristic(ctx, nbs);
        if (!Number.isFinite(h)) continue;
        const ng = g + 1;
        const nk = keyOf(nbs, b, nSet);
        if (nk === '') continue;
        if (best.has(nk) && best.get(nk)! <= ng) continue;
        best.set(nk, ng);
        const nid = sBoxes.length;
        sBoxes.push(nbs);
        sPlayer.push(b);
        sG.push(ng);
        sParent.push(id);
        sStep.push(walk + ch);
        heap.push([greedy ? h : ng + h, h, seq++, nid]);
      }
    }
  }
  return null;
}

/** The hallway row: the row whose cells are all floor except the borders. */
function findHallway(lv: Level): number {
  for (let y = 0; y < lv.h; y++) {
    let full = true;
    for (let x = 1; x < lv.w - 1; x++) {
      if (lv.wall[y * lv.w + x]) { full = false; break; }
    }
    if (full) return y;
  }
  return (lv.player / lv.w) | 0;
}

interface Room {
  boxes: number[];
  goals: number[];
  cells: Uint8Array;
  entry: number; // hallway cell outside this room's door
}

/** Split the level into rooms: floor components once the hallway row is cut. */
function findRooms(lv: Level, hallRow: number): Room[] {
  const { w, h } = lv;
  const open = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const y = (i / w) | 0;
    open[i] = !lv.wall[i] && y !== hallRow ? 1 : 0;
  }
  const comp = new Int32Array(w * h).fill(-1);
  const rooms: Room[] = [];
  for (let i = 0; i < w * h; i++) {
    if (!open[i] || comp[i] !== -1) continue;
    const cells = new Uint8Array(w * h);
    const region = flood(w, open, i);
    for (const c of region) { comp[c] = rooms.length; cells[c] = 1; }
    const boxes = lv.boxes.filter((b) => cells[b]);
    const goals = lv.goals.filter((g) => cells[g]);
    if (boxes.length === 0 && goals.length === 0) continue;
    // Entry: a hallway cell adjacent to a component cell (the door's
    // hallway square). Prefer the one nearest the room's first box.
    let entry = -1;
    let entryDoor = -1;
    for (const c of region) {
      const x = c % w;
      const y = (c / w) | 0;
      for (const [dx, dy] of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (ny !== hallRow || nx < 0 || nx >= w) continue;
        const hc = ny * w + nx;
        if (lv.wall[hc]) continue;
        if (entry === -1) { entry = hc; entryDoor = c; }
      }
    }
    if (entry === -1) continue; // no door onto the hallway: not a room
    void entryDoor;
    rooms.push({ boxes, goals, cells, entry });
  }
  // Deterministic order: above rooms left to right, then below rooms.
  const minPos = (cells: Uint8Array): [number, number] => {
    let minY = Infinity;
    let minX = Infinity;
    for (let i = 0; i < cells.length; i++) {
      if (!cells[i]) continue;
      const y = (i / w) | 0;
      const x = i % w;
      if (y < minY || (y === minY && x < minX)) { minY = y; minX = x; }
    }
    return [minY, minX];
  };
  rooms.sort((a, b) => {
    const [ay, ax] = minPos(a.cells);
    const [by, bx] = minPos(b.cells);
    return ay - by || ax - bx;
  });
  return rooms;
}

function solve(text: string): string {
  const lv = parse(text);
  const { w, h } = lv;
  const hallRow = findHallway(lv);
  const rooms = findRooms(lv, hallRow);

  // Sanity: every box and goal belongs to exactly one room.
  const covered =
    rooms.length > 0 &&
    rooms.every((r) => r.boxes.length === r.goals.length && r.boxes.length > 0) &&
    rooms.reduce((a, r) => a + r.boxes.length, 0) === lv.boxes.length &&
    rooms.reduce((a, r) => a + r.goals.length, 0) === lv.goals.length;

  const allFloor = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) allFloor[i] = lv.wall[i] ? 0 : 1;

  if (!covered) {
    // Fallback: one push-A* over the whole level.
    const ctx = makeRoomCtx(lv, allFloor, allFloor, lv.goals);
    const res = solveRoom(ctx, lv.boxes, lv.player, ROOM_BUDGET, false)
      ?? solveRoom(ctx, lv.boxes, lv.player, GREEDY_BUDGET, true);
    return res ? res.moves : '';
  }

  // Roam cells for a room: its cells plus the whole hallway row.
  const hallCells = new Uint8Array(w * h);
  for (let x = 0; x < w; x++) if (!lv.wall[hallRow * w + x]) hallCells[hallRow * w + x] = 1;

  let moves = '';
  let player = lv.player;
  const curBoxes = [...lv.boxes];
  for (const room of rooms) {
    // Walk from wherever the player stands to this room's entry hallway cell.
    const blocked = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) blocked[i] = lv.wall[i] ? 0 : 1;
    for (const b of curBoxes) blocked[b] = 0;
    const entryWalk = walkPath(w, h, blocked, player, room.entry);
    if (entryWalk === null) return '';
    moves += entryWalk;
    player = room.entry;

    const roam = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) roam[i] = room.cells[i] || hallCells[i] ? 1 : 0;
    const ctx = makeRoomCtx(lv, room.cells, roam, room.goals);
    const res = solveRoom(ctx, room.boxes, player, ROOM_BUDGET, false)
      ?? solveRoom(ctx, room.boxes, player, GREEDY_BUDGET, true);
    if (res === null) return '';
    moves += res.moves;
    // Replay the room's moves to learn the player's end cell and the new
    // box positions (needed for the next room's walks).
    const state = replay(lv, curBoxes, player, res.moves);
    if (state === null) return '';
    player = state.player;
    for (let i = 0; i < curBoxes.length; i++) curBoxes[i] = state.boxes[i];
  }
  return moves;
}

/** Replay `moves` on the level from the given boxes/player. Null when illegal. */
function replay(
  lv: Level,
  boxes0: number[],
  player0: number,
  moves: string,
): { boxes: number[]; player: number; pushes: number } | null {
  const boxes = [...boxes0];
  let player = player0;
  let pushes = 0;
  const boxAt = new Map<number, number>();
  boxes.forEach((b, i) => boxAt.set(b, i));
  for (const ch of moves) {
    const c = ch.toLowerCase();
    const d = DIRS.find(([, , k]) => k === c);
    if (!d) return null;
    const [dx, dy] = d;
    const x = player % lv.w;
    const y = (player / lv.w) | 0;
    const nx = x + dx;
    const ny = y + dy;
    if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) return null;
    const t = ny * lv.w + nx;
    if (lv.wall[t]) return null;
    const bi = boxAt.get(t);
    if (bi !== undefined) {
      const bx = nx + dx;
      const by = ny + dy;
      if (bx < 0 || bx >= lv.w || by < 0 || by >= lv.h) return null;
      const bt = by * lv.w + bx;
      if (lv.wall[bt] || boxAt.has(bt)) return null;
      boxAt.delete(t);
      boxAt.set(bt, bi);
      boxes[bi] = bt;
      pushes++;
    }
    player = t;
  }
  boxes.sort((a, b) => a - b);
  return { boxes, player, pushes };
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
