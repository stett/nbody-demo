# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

CMake-based, C++20, targeting Windows with MSVC. Dependencies (Cinder, BS thread-pool, Catch2) are git submodules.

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Run the executable from the build output directory. The Vulkan SDK is a hard build dependency — the GPU backend is always compiled. Whether a usable compute device exists is discovered at run time, and the simulation falls back to the CPU path when it doesn't.

## Architecture

This is an N-body gravitational simulation with interactive 3D visualization. It is split into two layers:

**`external/nbody/`** — self-contained physics library with no graphics dependencies:
- `include/nbody/sim.h` / `source/sim.cpp`: Main simulation orchestrator. `Sim::update()` calls `accelerate()` (builds the BH tree, then applies gravitational forces to each body in parallel via a BS thread pool) then `integrate()` (semi-implicit Euler with toroidal space wrapping).
- `include/nbody/bhtree.h` / `source/bhtree.cpp`: Barnes-Hut octree. `insert()` builds the tree recursively, computing center-of-mass at each node. `apply()` traverses it using the θ=0.5 opening angle criterion.
- `include/nbody/body.h`: 128-bit `Body` struct (position, velocity, acceleration, mass, radius) laid out for GPU compatibility.
- `source/gpu.h` / `source/gpu.cpp`: Vulkan RAII wrapper; manages pipelines for both N² and NLogN compute modes. Conditionally compiled.
- `source/util.cpp`: Galaxy (`disk()`) and uniform (`cube()`) initial condition generators with Keplerian orbital velocities.
- `include/nbody/constants.h`: Simulation constants (G=1, masses, max bodies ~1M).

**`source/`** — Cinder-based demo app:
- `demo.h` / `demo.cpp`: Cinder `App` subclass. Handles camera orbit, particle/wireframe rendering (geometry shaders), ImGui debug UI, mouse interaction for selecting and dragging bodies, and calling into `nbody::Sim` each frame.
- `main.cpp`: `CINDER_APP` entry point.

## Key Data Flow

```
spawn_galaxy() → Sim::update() [each frame]
                   ├─ accelerate(): build BH tree → parallel force summation (or Vulkan compute)
                   └─ integrate(): Euler step + toroidal wrap
                 → demo renders particles + BH tree wireframe + ImGui overlay
```
