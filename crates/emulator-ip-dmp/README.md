# DMP SMAPH rendering

The renderer consumes MMIO descriptors, addressed command lists and DMA arrays.
The `gles::Engine` implements the 3D data-before-header packet grammar and is
shared by the processor-specific bus adapters. VG path/fill command banks retain
their separate grammar.

3D lists upload vector instruction memory, operand descriptors, boolean/integer
controls and float uniforms. The vector interpreter executes transforms,
lighting and point setup from those instructions, including relative uniform
addressing, comparisons, calls, conditionals and bounded loops. Attributes come
from packed immediate vectors or DMA groups with explicit formats, strides,
padding slots and index buffers.

The raster pipeline supports triangles, strips, fans, lines and points; homogeneous
clipping; perspective interpolation; tiled color and depth/stencil targets;
two 2D texture units and combiners; nearest, bilinear and mip filtering; fog;
alpha/depth/stencil tests; color masks; and blending. Native pixmap transfers
accept byte pitches and preserve row padding. Unknown registers, instructions,
formats and unsupported configurations return errors. A rejected list publishes
no target writes or updated program state.

Native DMP 3D command lists use the host WGSL renderer by default and require a
hardware adapter. Set `FUJIMOD_DMP_CPU=1` to use the CPU renderer; WebAssembly
always uses the CPU renderer. `FUJIMOD_DMP_VERIFY_GPU=1` selects the GPU path
even when the CPU option is set, compares its snapshots and write ranges
against the CPU reference, then returns the GPU result. The CPU renderer uses
integer scan conversion with the same edge equations and top-left rule as the
host GPU path. Transcendentals use `libm`. Host GPU handles stay outside
emulated state and checkpoints. Diagnostic state reports the CPU rasterizer.

Instruction fields follow the vector arithmetic/flow layout described by
[nihstro's shader bytecode definitions](https://github.com/neobrain/nihstro/blob/a969b1a6849b02679a1858847ba18bedd3a19d44/include/nihstro/shader_bytecode.h).

Native SMAPH path rendering uses the shared GPU surfaces when a host adapter is
available. Set `FUJIMOD_SMAPH_CPU=1` to force CPU rendering. Set
`FUJIMOD_SMAPH_VERIFY_GPU=1` to compare GPU coverage and complete target bytes
against the CPU renderer while returning the CPU result.
Arithmetic intermediates use IEEE float32, primary color conversion truncates; combiners use integer eight-bit components,
ADD_SIGNED subtracts 128, and each scaled stage saturates separately. Blended
RGBA8 output rounds to the nearest byte,
and triangle coverage uses four fractional coordinate bits. Smooth lines use
an eight-step distance filter with asymmetric endpoint clipping; points remain
squares. The line filter is an empirical model of the sampled widths and angles. The interpreter
bounds instruction execution, call/loop nesting, uploads and addressed buffers.

Run the library regressions with:

```sh
cargo test --release --lib -p fujimod-emulator-ip-dmp
```

They cover program uploads, exact triangle coverage, CPU/GPU pixel parity,
shared edges and subpixels, checkpoint continuation between vertices, refusal
rollback, linked-list cycles, masked fills and pitched native copies.

## SMAPH verifier failure records

For native diagnostic runs, set `FUJIMOD_SMAPH_FAILURE_CAPTURE` to a fresh file
path together with `FUJIMOD_SMAPH_VERIFY_GPU=1`. On a paint mismatch the verifier
writes a CBOR record before returning its usual mismatch error. The parent
directory must exist; creation refuses an existing file. Capture I/O failures
are reported as errors. The option does not change the guest renderer inputs.

The version-1 record is a CBOR array containing, in order:

1. Schema version (`1`).
2. Failing target address.
3. First differing byte offset.
4. Path draws.
5. Fill draws and stream state.
6. Prepared paints, including texture and gradient state.
7. Coverage bounds for each draw.
8. GPU coverage words for each draw (already compared with CPU coverage).
9. Original guest targets, keyed by address, with width, height and bytes.
10. CPU reference targets in the same layout.
11. GPU bytes for the failing target.

The capture contains guest memory and belongs in local ignored research output.
It describes emulator execution; a successful capture does not qualify hardware
behavior or establish renderer parity.

SMAPH prepared solid colors are already transformed and packed into the target
byte order by draw preparation. GPU painting consumes them directly. Sampled
texture and gradient colors still pass through the packing stage. The CPU test
reference follows this same input contract.

For local failure analysis, the ignored Rust test
`captured_resident_batch_matches_cpu_per_draw` accepts a version-1 failure record
through `FUJIMOD_SMAPH_REPLAY_CAPTURE`. Run it explicitly with `--ignored
--nocapture` and a working GPU adapter. It compares the complete target after
each draw, reporting the first differing draw and byte without firmware startup.

`FUJIMOD_SMAPH_COVERAGE_CAPTURE_DIR` records each nonempty native coverage
batch before GPU submission. The directory must already exist; numbered files
include the process ID and refuse replacement. Each buffered CBOR file contains
schema version 1 followed by a list of `(bounds, samples, edges, stroke_radius,
nonzero)` records. The last submitted batch can therefore be inspected even if
the GPU fence has not completed. Files are local diagnostic output.

Large native SMAPH coverage paths (more than 256 edges) are split into eight-row
bands before GPU dispatch. Each band retains edges whose vertical range can
intersect its samples. Stroke ranges include the stroke radius and retain
zero-length segments. Fill paths omit horizontal edges because they contribute
no crossing. GPU counts are copied back into their original row-major positions;
requests are dispatched in batches of at most 256. The captured large-path
regression compares every output count with CPU coverage.
