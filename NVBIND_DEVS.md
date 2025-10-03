# NVBIND Integration Status & Questions (September 2025)

## Quick refresher
- Re-reviewed the vendored `archive/nvbind` tree after the most recent upstream sync. The layout now ships both the legacy `cdi` module **and** a richer `cdi_v07` implementation covering CDI 0.7+ constructs (topology, dynamic allocation, hot-plug, richer `ContainerEdits`, etc.).
- The Bolt runtime currently calls `nvbind::GpuManager::{generate_gaming_cdi_spec, generate_aiml_cdi_spec, generate_default_cdi_spec}` and funnels the result through our own `CdiSpec` shim in `src/runtime/gpu_integration.rs`.
- That shim only preserves three things (`Vec<String>` of device node paths, mount paths, hook identifiers). We throw away structured data that now exists in upstream (env vars, per-hook metadata, mount host vs container paths, mount options, device majors/minors, annotations, etc.).

## Where we are blocked/confused
1. **Hook wiring:**
   - Upstream `archive/nvbind/src/cdi/bolt.rs` (and the CDI v0.7 module) now emits hooks with full metadata: name, executable path, args, env, timeout.
   - Bolt’s `AppliedCdiSpec` keeps hooks as bare strings, so we cannot legally translate them into OCI lifecycle entries. We also don’t know whether the string we receive today is a file path, an alias, or a serialized JSON blob.
   - Question: *What is the canonical representation of hooks that nvbind intends Bolt to consume?* Should we expect the async API to expose a typed struct (similar to `Hook` in the nvbind repo) or should we parse JSON ourselves?

2. **Mount semantics:**
   - CDI v0.6/0.7 exposes `Mount { host_path, container_path, options }`, but our shim collapses them into a single string. When we inject mounts into the OCI spec we currently bind the same host path to the identical in-container path with fixed options (`bind,ro,nosuid,nodev`).
   - We need to know if nvbind expects us to respect `container_path` (which may differ from host path) and any specific mount options.
   - Question: *Can we assume nvbind will always hand back explicit host/container pairs and option lists?* If so, we should mirror those exactly in the OCI spec instead of guessing.

3. **Device nodes & metadata:**
   - CDI v0.7 device entries carry majors/minors, file modes, annotations, and topology hints. Our shim only relays the string path, so we have to `stat` the device to recover major/minor and we lose everything else.
   - Potential issue: if nvbind starts shipping synthetic devices (e.g., MIG partitions) the major/minor may not match the on-host node. We need to know whether the shim should ingest the serialized `DeviceNode` entries directly.

4. **Environment propagation:**
   - Upstream CDI specs insert GPU-specific env vars (driver capabilities, workload type, MIG settings, etc.). Bolt currently ignores them entirely. Some are critical for CUDA to pick up the right devices.
   - Question: *Should Bolt read the `container_edits.env` field and append it to our OCI process env list?* If yes, we need clarity on merge strategy (override vs append, precedence when the container already defines the same key).

5. **API surface expectations:**
   - The git dependency still compiles because the exported Rust type has not changed (it serializes down into our minimal struct via `serde`). However, we now understand that we are silently discarding fields.
   - We want to confirm whether the long-term plan is to expose a stable public Rust struct for Bolt (e.g., via the `nvbind::bolt` module) that already contains ergonomic helpers for OCI runtimes.

## Proposed next steps (pending guidance)
- Replace our local `CdiSpec` shim with a data model that mirrors nvbind’s `ContainerEdits`, `DeviceNode`, `Mount`, and `Hook` structs so we don’t lose information.
- Extend `AppliedCdiSpec` to store structured hooks, mounts, device nodes, and env vars, and teach `create_oci_spec` to translate each piece into `oci_spec::runtime::{Hook, Mount, LinuxDevice}` entries without guessing defaults.
- Add unit tests that deserialize sample nvbind CDI specs (gaming / AIML / default) and assert we emit the exact OCI wiring expected (hooks in the right lifecycle phase, mounts deduped, env merged).
- Only after the above, tackle the optional CDI v0.7 extras (topology, dynamic allocation, annotations) so we can future-proof scheduling logic.

## Requests for the nvbind team
- **Documentation/example:** a minimal “hello world” JSON spec (or Rust fixture) showing how gaming/AIML hooks are expected to execute, including any required environment side-effects.
- **API confirmation:** whether we should rely on the `nvbind::cdi::CdiSpec` struct, or if a Bolt-specific helper is planned to avoid tight coupling to internal layout.
- **Hook lifecycle guidance:** clarify which OCI hook phases you expect us to use (`prestart`, `poststart`, `prestop`, others?) and any ordering guarantees.
- **Mount dedupe policy:** do you expect us to deduplicate or preserve duplicate paths exactly as emitted? Some specs include overlapping library directories.

Happy to adjust approach once we hear back—right now we’re hesitant to implement the OCI hook propagation until we understand the canonical data we should consume.
