# Sokoban solver

Push-optimal Sokoban solver for make.fun's `sokoban-solver-01-run-20261121`
bounty. `main.ts` (Node 22, no dependencies) reads `/input/<id>/level.txt`
for each instance of a set and writes `/output/<id>/solution.txt`.

Method: the level is a one-square-wide hallway with up to four rooms off
it (one door each, 3-4 boxes and as many goals per room), and in an optimal
solution no box ever leaves its room. The solver splits the level into
rooms (connected floor components once the hallway row is removed), solves
each room independently with push-based A* (admissible min-cost box-goal
assignment heuristic, dead-square and 2x2-block pruning), and stitches the
rooms with BFS hallway walks. Deterministic: budgets are counts of expanded
states, never wall time, so two runs produce identical bytes.

Local run:

```
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Practice set (push-optimal counts 13/16/29/47/60): 1,000,000 points.
