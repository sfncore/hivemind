# Hivemind Runtime Internals

> Reference for Node0 integration. Pinned to commit `4d5c414`.
> All file:line references are relative to the repository root.

---

## Table of Contents

1. [Server / Runtime / ModuleBackend](#1-server--runtime--modulebackend)
2. [DHT — Peer Discovery & Health Monitoring](#2-dht--peer-discovery--health-monitoring)
3. [DecentralizedAverager — Gradient & State Averaging](#3-decentralizedaverager--gradient--state-averaging)
4. [Collaborative Optimizer](#4-collaborative-optimizer)
5. [libp2p Networking](#5-libp2p-networking)
6. [Matchmaking & Group Formation](#6-matchmaking--group-formation)
7. [Expert Routing — Batch Routing Between Pipeline Stages](#7-expert-routing--batch-routing-between-pipeline-stages)

---

## 1. Server / Runtime / ModuleBackend

The training loop execution engine. The Server orchestrates all components; Runtime processes
batches; ModuleBackend wraps PyTorch modules for distributed forward/backward.

### 1.1 Server (`hivemind/moe/server/server.py`)

**Class:** `Server` (line 35)

Orchestrates module backends, connection handlers, runtime, DHT integration, and checkpointing.

| Method | Line | Purpose |
|--------|------|---------|
| `__init__` | 55 | Initialize server with backends, handlers, runtime, DHT handler |
| `create` (classmethod) | 88 | Factory: load experts, create optimizers, instantiate backends |
| `run` | 231 | Start server in current thread (blocking) |
| `run_in_background` | 258 | Start server in background thread |
| `shutdown` | 280 | Gracefully terminate all components |

**Startup sequence** (lines 231–256):
1. Log expert info
2. Start DHT if not running
3. Start `DHTHandlerThread` — periodically declares experts to DHT
4. Start `CheckpointSaver` if enabled
5. Start all `ConnectionHandler` processes
6. Run `Runtime` (blocks until shutdown)

### 1.2 Runtime (`hivemind/moe/server/runtime.py`)

**Class:** `Runtime` (line 22)

The main batch processing loop. Selects batches from task pools by priority, executes them
on the target device, and dispatches results.

| Method | Line | Purpose |
|--------|------|---------|
| `__init__` | 48 | Initialize with backends and configuration |
| `run` | 68 | Main execution loop |
| `iterate_minibatches_from_pools` | 133 | Select & load batches with priority ordering |

**Training loop** (lines 68–106):
```
1. Start all task pools
2. Move experts to device
3. Create ThreadPool for async output sending
4. FOR EACH batch (selected by earliest timestamp priority):
   a. pool.process_batch() → executes forward or backward
   b. Send outputs asynchronously via ThreadPool
```

**Batch selection** (lines 133–155): Uses `DefaultSelector` to monitor all task pool batch
receivers. Selects pool with lowest priority value (earliest timestamp = processed first).

### 1.3 ModuleBackend (`hivemind/moe/server/module_backend.py`)

**Class:** `ModuleBackend` (line 19)

Wraps a PyTorch `nn.Module` for distributed training. Manages forward/backward execution
and optimizer updates.

| Method | Line | Purpose |
|--------|------|---------|
| `__init__` | 45 | Initialize backend with module, optimizer, scheduler, schemas |
| `forward` | 83 | Process batch of forward requests (`torch.no_grad()`) |
| `backward` | 106 | Process batch of backward requests + optimizer step |
| `on_backward` | 156 | Run `optimizer.step()` and `scheduler.step()` |

**Key design:** Backward **re-runs forward** with `enable_grad()` (line 135) then calls
`torch.autograd.backward()` (line 147). This allows stateless operation — Runtime doesn't
need to guarantee forward/backward ordering. Similar to gradient checkpointing.

### 1.4 TaskPool (`hivemind/moe/server/task_pool.py`)

**Class:** `TaskPool` (line 59)

Aggregates individual requests into batches for GPU-efficient processing.

| Method | Line | Purpose |
|--------|------|---------|
| `submit_task` | 102 | Queue task, return Future |
| `iterate_minibatches` | 113 | Aggregate tasks into batches (respects min/max batch size, timeout) |
| `_pool_input_loop` | 161 | Concatenate task args, send to Runtime |
| `_pool_output_loop` | 194 | Split batch outputs, dispatch to individual task futures |

**Priority:** Lower timestamp = processed sooner. Ensures FIFO with timeout fairness.

### 1.5 ConnectionHandler (`hivemind/moe/server/connection_handler.py`)

**Class:** `ConnectionHandler` (line 22)

Receives RPC requests, deserializes tensors, submits to task pools, serializes results.

| Method | Line | Purpose |
|--------|------|---------|
| `rpc_forward` | 134 | Handle single forward request |
| `rpc_forward_stream` | 142 | Handle streaming forward requests |
| `rpc_backward` | 156 | Handle single backward request |
| `rpc_backward_stream` | 165 | Handle streaming backward requests |

### 1.6 DHTHandlerThread (`hivemind/moe/server/dht_handler.py`)

**Class:** `DHTHandlerThread` (line 22)

Periodically declares experts to DHT so clients can discover them.

| Function | Line | Purpose |
|----------|------|---------|
| `declare_experts` | 41 | Store `uid → peer_id` mappings in DHT |
| `get_experts` | 81 | Retrieve expert info from DHT by UID |

### 1.7 Request/Response Flow

```
Client RPC → ConnectionHandler.rpc_forward() [line 134]
  → Deserialize tensors
  → ModuleBackend.forward_pool.submit_task() [line 102]
  → TaskPool._pool_input_loop() batches tasks [line 161]
  → Runtime selects pool, calls process_batch() [line 110]
  → ModuleBackend.forward() executes module [line 101]
  → TaskPool._pool_output_loop() splits outputs [line 194]
  → ConnectionHandler serializes & returns
```

### 1.8 Checkpoints (`hivemind/moe/server/checkpoints.py`)

| Function | Line | Purpose |
|----------|------|---------|
| `store_experts` | 53 | Save all expert state_dicts to timestamped files |
| `load_experts` | 67 | Load experts from `checkpoint_last.pt` symlinks |

---

## 2. DHT — Peer Discovery & Health Monitoring

Kademlia-style distributed hash table for peer discovery, key-value storage, and health
monitoring.

### 2.1 DHT (`hivemind/dht/dht.py`)

**Class:** `DHT` (line 22)

High-level interface running as a background `mp.ForkProcess`. Provides `get()` and
`store()` via inter-process pipes.

| Method | Line | Purpose |
|--------|------|---------|
| `run` | 89 | Main event loop for DHT process |
| `get` | 166 | External GET interface via IPC |
| `store` | 192 | External STORE interface via IPC |
| `add_validators` | 270 | Register custom record validators |

### 2.2 DHTNode (`hivemind/dht/node.py`)

**Class:** `DHTNode` (line 45)

Core asyncio-based DHT node. One per process.

| Method | Line | Purpose |
|--------|------|---------|
| `create` (classmethod) | 99 | Factory with bootstrap (ping initial peers, self-discovery) |
| `find_nearest_nodes` | 278 | Beam search for k-nearest nodes by DHTID |
| `store_many` | 351 | Store with replication to `num_replicas` nearest nodes |
| `get_many_by_id` | 569 | Multi-key retrieval with DHT traversal |
| `_refresh_routing_table` | 796 | Background: query random DHTID in stale buckets |
| `_refresh_stale_cache_entries` | 727 | Background: refresh cached values before expiration |

**Bootstrap** (lines 219–260):
1. Ping all `initial_peers`, wait for first response
2. Gather remaining peers within `bootstrap_timeout`
3. Self-discovery: `find_nearest_nodes([self.node_id])` to populate routing table

### 2.3 Routing Table (`hivemind/dht/routing.py`)

**Class:** `RoutingTable` (line 20) — Kademlia bucket tree.

| Method | Line | Purpose |
|--------|------|---------|
| `add_or_update_node` | 49 | Add node, split buckets if needed |
| `get_nearest_neighbors` | 109 | Heap-based k-nearest search |

**Class:** `KBucket` (line 167) — Single bucket `[lower, upper)`.

| Method | Line | Purpose |
|--------|------|---------|
| `add_or_update_node` | 185 | Add or keep as replacement if full |
| `split` | 233 | Split into left/right at midpoint |

**Class:** `DHTID` (line 252) — 160-bit SHA1-based node identifier.

| Method | Line | Purpose |
|--------|------|---------|
| `generate` | 263 | Generate from source or random |
| `xor_distance` | 275 | Kademlia XOR distance metric |

### 2.4 Protocol (`hivemind/dht/protocol.py`)

**Class:** `DHTProtocol` (line 25)

| Method | Line | Purpose |
|--------|------|---------|
| `call_ping` | 97 | Ping peer, update routing table |
| `call_store` | 164 | Store key-value pairs on peer |
| `call_find` | 271 | Request keys + k-nearest peers |
| `update_routing_table` | 371 | Update table after any RPC |

### 2.5 Traversal (`hivemind/dht/traverse.py`)

**Function:** `traverse_dht` (line 72)

Multi-query beam search with concurrent workers. Complexity: O(log N × k) RPCs.

Algorithm:
1. Maintain min-heaps of candidate nodes per query (by XOR distance)
2. Workers select least-explored query, pop nearest candidate
3. Call `get_neighbors(peer, queries)`, update heaps
4. Terminate when beam_size nearest found or no more candidates

### 2.6 Storage (`hivemind/dht/storage.py`)

**Class:** `DHTLocalStorage` (line 35) — Local node storage.

| Method | Line | Purpose |
|--------|------|---------|
| `store` | 38 | Store regular value or delegate to subkey |
| `store_subkey` | 51 | Store/update subkey in DictionaryDHTValue |

**Class:** `DictionaryDHTValue` (line 11) — Maps subkeys to values with individual expirations.

### 2.7 Health Monitoring

**Blacklist** (`hivemind/dht/node.py:897`):
- `register_failure()` (line 909): Exponential backoff ban (`base_time × backoff^fail_count`)
- `register_success()` (line 916): Remove from blacklist, reset counter

**Routing table refresh** (`hivemind/dht/node.py:796`): Periodic task queries random DHTID
in stale buckets every `refresh_timeout` seconds.

**Cache refresh** (`hivemind/dht/node.py:727`): Proactively refreshes cached values
`cache_refresh_before_expiry` seconds before expiration.

### 2.8 Validation & Crypto

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `RecordValidatorBase` | `dht/validation.py` | 14 | Abstract record validator |
| `SchemaValidator` | `dht/schema.py` | 15 | Pydantic schema enforcement |
| `RSASignatureValidator` | `dht/crypto.py` | 12 | RSA signature protection |

---

## 3. DecentralizedAverager — Gradient & State Averaging

### 3.1 DecentralizedAverager (`hivemind/averaging/averager.py`)

**Class:** `DecentralizedAverager` (line 50)

Runs as a background `mp.Process`. Periodically averages tensors with peers via matchmaking
+ all-reduce.

| Method | Line | Purpose |
|--------|------|---------|
| `step` | 367 | User-facing: trigger one averaging round |
| `_step` | 421 | Main control: matchmaking → allreduce |
| `_aggregate_with_group` | 514 | Compute peer fractions from bandwidths, run allreduce |
| `_run_allreduce_inplace_` | 537 | Execute AllReduceRunner, update local tensors |
| `get_tensors` | 564 | Thread-safe context manager for averaged tensors |
| `rpc_join_group` | 574 | RPC: handle join requests (delegates to Matchmaking) |
| `rpc_aggregate_part` | 581 | RPC: handle tensor aggregation (delegates to AllReduceRunner) |
| `load_state_from_peers` | 668 | Download optimizer state from best-priority peer |

### 3.2 AllReduceRunner (`hivemind/averaging/allreduce.py`)

**Class:** `AllReduceRunner` (line 32)

Butterfly all-reduce in a predefined peer group. Returns deltas (not absolutes) for
numerical stability.

| Method | Line | Purpose |
|--------|------|---------|
| `run` | 151 | Main async generator yielding averaged tensor deltas |
| `_communicate_with_peer` | 201 | Send local parts, receive averaged results |
| `rpc_aggregate_part` | 259 | RPC: receive peer's parts, accumulate, return averages |
| `_accumulate_parts_streaming` | 335 | Core: deserialize → accumulate → serialize deltas |

**AveragingMode** (line 26):
- `NODE` (0): Full participant
- `CLIENT` (1): Downloads but doesn't aggregate
- `AUX` (2): Assists without sending local tensors

### 3.3 Partition (`hivemind/averaging/partition.py`)

**Class:** `TensorPartContainer` (line 21) — Splits tensors by peer_fractions.

| Method | Line | Purpose |
|--------|------|---------|
| `__init__` | 35 | Split tensors proportionally to peer fractions |
| `iterate_input_parts_for` | 104 | Compressed parts for a specific peer |
| `iterate_output_tensors` | 138 | Reconstruct averaged tensors from received parts |

**Class:** `TensorPartReducer` (line 179) — Accumulates parts from all senders.

| Method | Line | Purpose |
|--------|------|---------|
| `accumulate_part` | 218 | Add weighted part, return average when all received |
| `on_sender_failed` | 248 | Ban sender, exclude future parts |

### 3.4 Load Balancing (`hivemind/averaging/load_balancing.py`)

**Function:** `load_balance_peers` (line 13)

Computes optimal peer fractions via linear programming. Minimizes
`max_i(communication_i / bandwidth_i)` — the slowest peer determines round time.

**Function:** `hagenbach_bishoff` (line 89) — Converts continuous fractions to discrete
element counts that sum exactly to `vector_size`.

### 3.5 Control (`hivemind/averaging/control.py`)

**Class:** `StepControl` (line 22) — Cross-process synchronization via shared torch tensor
(18 bytes).

**Stages:** `IDLE → LOOKING_FOR_GROUP → AWAITING_TRIGGER → RUNNING_ALLREDUCE → FINISHED`

Shared buffer layout:
```
[0:8]  scheduled_time (double)
[8:16] weight (double)
[16]   stage (AveragingStage, 1 byte)
[17]   began_allreduce (bool, 1 byte)
```

---

## 4. Collaborative Optimizer

### 4.1 Optimizer (`hivemind/optim/optimizer.py`)

**Class:** `Optimizer` (line 32)

Wraps any PyTorch optimizer for collaborative training. Coordinates gradient accumulation,
averaging, and parameter synchronization.

| Method | Line | Purpose |
|--------|------|---------|
| `__init__` | 167 | Initialize with local params, averagers, tracker |
| `step` | 369 | Accumulate grads, trigger averaging at epoch boundary |
| `_update_global_epoch` | 438 | Gradient averaging → optimizer step → state averaging |
| `_begin_averaging_gradients` | 511 | Start all-reduce for gradient averaging |
| `load_state_from_peers` | 679 | Download params + optimizer state from peers |

**Key configuration:**
- `target_batch_size`: Global batch size before optimizer step
- `batch_size_per_step`: Local batch size per call
- `offload_optimizer`: Run optimizer on CPU (separate from GPU model)
- `delay_optimizer_step` / `delay_grad_averaging`: Background execution
- `use_local_updates`: Update params every step + average periodically (delta rule)

**Training flow:**
```
Optimizer.step(batch_size):
  1. Accumulate gradients locally
  2. Report progress to tracker
  3. Pre-schedule averaging if epoch approaching
  4. IF ready_to_update_epoch:
     a. Average gradients with peers
     b. Load averaged grads into optimizer
     c. optimizer.step() via state_averager
     d. Average parameters with peers (every N epochs)
     e. Reset accumulators, advance epoch
```

### 4.2 GradientAverager (`hivemind/optim/grad_averager.py`)

**Class:** `GradientAverager` (line 18) — extends `DecentralizedAverager`

| Method | Line | Purpose |
|--------|------|---------|
| `accumulate_grads_` | 130 | Add current gradients to local accumulators (scaled by batch size) |
| `step` | 163 | All-reduce accumulated gradients with peers |
| `use_averaged_gradients` | 223 | Context manager: substitute model grads with averaged grads |

Manages three gradient buffers: model grads (device), accumulators (CPU), averaged (CPU shared).

### 4.3 TrainingStateAverager (`hivemind/optim/state_averager.py`)

**Class:** `TrainingStateAverager` (line 37) — extends `DecentralizedAverager`

| Method | Line | Purpose |
|--------|------|---------|
| `step` | 329 | Optimizer step + scheduler step + state averaging |
| `_apply_averaging_results_` | 606 | Copy averaged tensors back (with delta rule support) |
| `get_current_state` | 627 | Serialize state for new peers |
| `load_state_from_peers` | 658 | Download and synchronize state |

**Delta rule** (lines 614–621): When `use_local_updates=True`, computes
`delta = new_averaged - old_averaged` and adds to local tensors. Preserves local updates
while still synchronizing globally.

### 4.4 PowerSGD (`hivemind/optim/power_sgd_averager.py`)

**Class:** `PowerSGDGradientAverager` (line 28) — Gradient compression via rank-r factorization.

| Method | Line | Purpose |
|--------|------|---------|
| `_aggregate_with_group` | 132 | Two-phase all-reduce: Phase P (orthogonalize), Phase Q (reconstruct) |

Two-phase all-reduce:
1. All-reduce P matrices (m × r), orthogonalize
2. All-reduce Q matrices (n × r) + uncompressed gradients
3. Reconstruct: `new_m = P @ Q^T` with error feedback

### 4.5 ProgressTracker (`hivemind/optim/progress_tracker.py`)

**Class:** `ProgressTracker` (line 44) — Background thread tracking/reporting progress.

| Method | Line | Purpose |
|--------|------|---------|
| `report_local_progress` | 153 | Report samples accumulated, update throughput EMA |
| `_progress_reporter` | 195 | Background: publish `LocalTrainingProgress` to DHT |
| `_progress_fetcher` | 235 | Background: fetch global progress, calculate ETA |

**Epoch transition condition** (line 128–134):
```python
ready = (global_epoch > local_epoch           # another peer advanced
         or samples >= target_batch_size      # enough global samples
         or time >= eta_next_epoch)           # timeout
```

### 4.6 GradScaler (`hivemind/optim/grad_scaler.py`)

**Class:** `GradScaler` (line 25) — Custom `torch.amp.GradScaler` for `reuse_grad_buffers=True`.

Defers unscaling until global optimizer steps. Handles offloaded optimizer states.

---

## 5. libp2p Networking

### 5.1 P2P Daemon (`hivemind/p2p/p2p_daemon.py`)

**Class:** `P2P` (line 41)

Manages a libp2p daemon (p2pd) subprocess. The daemon handles NAT traversal, connection
management, and stream multiplexing.

| Method | Line | Purpose |
|--------|------|---------|
| `create` (classmethod) | 83 | Spawn p2pd process, wait for ready, establish client |
| `replicate` (classmethod) | 293 | Connect to existing daemon (multi-client) |
| `shutdown` | 641 | Kill daemon, clean up Unix sockets |
| `get_visible_maddrs` | 319 | Get publicly reachable multiaddresses |
| `generate_identity` | 279 | Generate RSA key pair for peer identity |
| `is_identity_taken` | 248 | Check if PeerID exists in swarm |

**Daemon configuration** (lines 82–109):
- `dht_mode`: "server"/"client"/"auto"
- `auto_nat`: Enable AutoNAT for reachability testing
- `nat_port_map`: UPnP/NAT-PMP port mapping
- `use_relay` / `use_auto_relay`: Circuit relay for NAT traversal
- `tls`: TLS 1.3 encryption
- `idle_timeout`: Kill idle daemon
- `relay_hop_limit`: Max relay chain depth

**Message protocol:**
- 1-byte marker: `0x00` (success) or `0x01` (error)
- 8-byte big-endian length header
- Payload in 64KB chunks

### 5.2 RPC Framework (`hivemind/p2p/servicer.py`)

**Class:** `ServicerBase` (line 33)

| Method | Line | Purpose |
|--------|------|---------|
| `_collect_rpc_handlers` | 48 | Scan `rpc_*` methods, extract type hints, build handler list |
| `add_p2p_handlers` | 107 | Register all RPC handlers on P2P daemon |
| `get_stub` | 141 | Generate client stub for calling remote servicers |

Handler naming: `{namespace}::{ClassName}.{method_name}`

### 5.3 Daemon Bindings (`hivemind/p2p/p2p_daemon_bindings/`)

**`control.py`** — Communication with p2pd over Unix sockets.

| Class/Method | Line | Purpose |
|--------------|------|---------|
| `DaemonConnector.open_connection` | 57 | Open Unix or TCP socket to daemon |
| `DaemonConnector.open_persistent_connection` | 68 | Upgrade for multiplexed unary RPCs |
| `ControlClient.add_unary_handler` | 255 | Register unary handler on daemon |
| `ControlClient.call_unary_handler` | 286 | Call remote unary handler |
| `ControlClient.stream_open` | 370 | Open bidirectional stream to peer |

**`datastructures.py`** — Core P2P data types.

| Class | Line | Purpose |
|-------|------|---------|
| `PeerID` | 18 | SHA256 multihash of public key, base58 encoding |
| `StreamInfo` | 97 | Stream metadata (peer_id, addr, proto) |
| `PeerInfo` | 116 | Peer identity + multiaddresses |

### 5.4 NAT Traversal

| Mechanism | Config | How it works |
|-----------|--------|--------------|
| **AutoNAT** | `auto_nat=True` | Tests if node is publicly reachable |
| **UPnP/NAT-PMP** | `nat_port_map=True` | Opens ports on home router |
| **Circuit Relay** | `use_relay=True` | Routes through relay nodes |
| **AutoRelay** | `use_auto_relay=True` | Discovers relays automatically |
| **Force Reachability** | `force_reachability="public"/"private"` | Override detection |

### 5.5 Protocol Buffers

| Proto File | Service | Key Messages |
|------------|---------|--------------|
| `proto/p2pd.proto` | Daemon control | Request, Response, PersistentConnectionRequest, StreamOpenRequest |
| `proto/runtime.proto` | MoE experts | ExpertRequest, ExpertResponse, Tensor (with CompressionType) |
| `proto/averaging.proto` | Averaging | JoinRequest, MessageFromLeader, AveragingData |
| `proto/dht.proto` | DHT | PingRequest/Response, StoreRequest/Response, FindRequest/Response |
| `proto/auth.proto` | Authentication | RequestAuthInfo, ResponseAuthInfo |

---

## 6. Matchmaking & Group Formation

### 6.1 Matchmaking (`hivemind/averaging/matchmaking.py`)

**Class:** `Matchmaking` (line 23)

Leader-follower protocol for forming averaging groups.

| Method | Line | Purpose |
|--------|------|---------|
| `look_for_group` | 111 | Main entry: run matchmaking with timeout |
| `_request_join_potential_leaders` | 148 | Try leaders in priority order |
| `_request_join_group` | 177 | Send JoinRequest to leader, handle accept/reject |
| `rpc_join_group` | 261 | RPC: handle incoming join requests from followers |
| `leader_assemble_group` | 370 | Create group_id, shuffle peers, gather metadata |
| `follower_assemble_group` | 390 | Build group from leader's message |

**Protocol flow:**
```
1. Non-client peers declare themselves in DHT under group key
2. Followers poll DHT for potential leaders (sorted by expiration time)
3. Follower sends JoinRequest → Leader validates → ACCEPTED/REJECTED
4. Leader assembles group when enough peers join
5. Leader sends BEGIN_ALLREDUCE with group_id, ordered_peer_ids
6. All peers begin all-reduce
```

**Leader priority:** Higher expiration time wins. Tie-breaker: lexicographic PeerID.

### 6.2 GroupKeyManager (`hivemind/averaging/key_manager.py`)

**Class:** `GroupKeyManager` (line 22)

Manages DHT keys for peer discovery. Key format: `{prefix}.0b{group_bits}`.

| Method | Line | Purpose |
|--------|------|---------|
| `declare_averager` | 46 | Register peer at group key in DHT |
| `get_averagers` | 70 | Query DHT for peers at group key |
| `update_key_on_group_assembled` | 94 | Deterministic bit update after successful averaging |

**Group bits update** (lines 94–105): After group assembly, uses `group_id` as RNG seed to
deterministically shuffle indices, then appends new bits (rolling buffer). This creates a
hierarchical key structure enabling logarithmic convergence.

### 6.3 PotentialLeaders (`hivemind/averaging/matchmaking.py`)

**Class:** `PotentialLeaders` (line 413)

| Method | Line | Purpose |
|--------|------|---------|
| `begin_search` | 429 | Start DHT queries + periodic self-declaration |
| `pop_next_leader` | 468 | Get most-suitable leader by expiration priority |
| `_update_queue_periodically` | 499 | Refresh leader list from DHT |

### 6.4 GroupInfo (`hivemind/averaging/group_info.py`)

**Dataclass:** `GroupInfo` (line 7)
- `group_id: bytes` — Random unique ID from leader
- `peer_ids: Tuple[PeerID, ...]` — Ordered peers for partition assignment
- `gathered: Tuple[bytes, ...]` — Serialized metadata from all peers

---

## 7. Expert Routing — Batch Routing Between Pipeline Stages

### 7.1 RemoteMixtureOfExperts (`hivemind/moe/client/moe.py`)

**Class:** `RemoteMixtureOfExperts` (line 25)

Main MOE layer. Routes inputs to top-k experts via beam search.

| Method | Line | Purpose |
|--------|------|---------|
| `forward` | 77 | Project → beam search → parallel expert calls → weighted sum |
| `compute_expert_scores` | 141 | Sum grid dimension scores for each expert |

**Forward flow:**
1. Project input → grid scores (line 95)
2. Beam search for `k_best` experts per sample (line 98)
3. `_RemoteCallMany.apply()` — dispatch to all experts in parallel (line 114)
4. Weight outputs by softmax of expert logits (line 130–137)

**Fault tolerance:** `k_min` experts must succeed; others are masked out. The mask propagates
through backward pass.

### 7.2 Beam Search (`hivemind/moe/client/beam_search.py`)

**Class:** `MoEBeamSearcher` (line 27)

| Method | Line | Purpose |
|--------|------|---------|
| `find_best_experts` | 233 | Main entry: single-batch expert selection |
| `batch_find_best_experts` | 354 | Batch version for multiple samples |
| `get_initial_beam` | 97 | Top-k active prefixes for first grid dimension |
| `get_active_successors` | 169 | Valid experts for a set of prefixes |

**Algorithm** (`_find_best_experts`, line 264):
1. Get initial beam for first dimension from DHT
2. For each subsequent dimension:
   a. Collect matching experts, keep top `beam_size` in max-heap
   b. Form new beam from successors with highest combined scores
   c. Fetch successors from DHT
3. Return top-k experts by total score

Complexity: O(k × D × S) where D=dimensions, S=grid_size per dimension.

### 7.3 RemoteExpert (`hivemind/moe/client/expert.py`)

**Class:** `RemoteExpert` (line 32)

| Method | Line | Purpose |
|--------|------|---------|
| `forward` | 60 | Execute expert on remote server (autograd-compatible) |

**Class:** `_RemoteModuleCall` (line 194) — Autograd function.

| Method | Line | Purpose |
|--------|------|---------|
| `forward` | 198 | Serialize inputs, send via RPC, return outputs |
| `backward` | 220 | Send grad_outputs via RPC, return grad_inputs |

**Streaming vs unary:** Automatically selects streaming gRPC for payloads exceeding
`MAX_UNARY_PAYLOAD_SIZE` (line 188).

### 7.4 Switch MOE (`hivemind/moe/client/switch_moe.py`)

**Class:** `RemoteSwitchMixtureOfExperts` (line 17)

Sparse routing variant: `k_best=1` (one expert per sample).

| Feature | Line | Purpose |
|---------|------|---------|
| Multiplicative jitter | 78 | Noise regularization |
| Grid dropout | 84 | Drop grid indices during training |
| Utilization EMA | 131 | Track expert usage balance |
| Load balancing loss | 151 | Auxiliary loss to equalize utilization |

### 7.5 Compression (`hivemind/compression/`)

| Class | File | Line | Strategy |
|-------|------|------|----------|
| `NoCompression` | `base.py` | 79 | Pass-through (numpy tobytes) |
| `Float16Compression` | `floating.py` | 10 | fp32→fp16 with clamping |
| `ScaledFloat16Compression` | `floating.py` | 43 | Mean-std normalize + fp16 |
| `Uniform8BitQuantization` | `quantization.py` | 60 | Uniform 256-bin quantization |
| `Quantile8BitQuantization` | `quantization.py` | 77 | Empirical quantile boundaries |
| `BlockwiseQuantization` | `quantization.py` | 130 | bitsandbytes 4096-element blocks |

**Adaptive selection** (`adaptive.py`):
- `SizeAdaptiveCompression` (line 25): Switch by tensor numel
- `RoleAdaptiveCompression` (line 35): Switch by TensorRole (ACTIVATION/PARAMETER/GRADIENT/OPTIMIZER)
- `PerTensorCompression` (line 59): Manual per-tensor mapping

---

## Appendix: Key Utility Modules

### Public API (`hivemind/__init__.py`)

Exports: `DHT`, `Server`, `ModuleBackend`, `RemoteExpert`, `RemoteMixtureOfExperts`,
`RemoteSwitchMixtureOfExperts`, `DecentralizedAverager`, `Optimizer`, `GradScaler`,
`TrainingAverager`, `P2P`, `P2PContext`, `PeerID`, `PeerInfo`, `register_expert_class`.

Version: `1.2.0.dev0` (line 16).

### MPFuture (`hivemind/utils/mpfuture.py`)

**Class:** `MPFuture` (line 65)

Cross-process future using shared memory. Origin process awaits; child process sets result
via shared pipe. Uses `SharedBytes` (line 31) for process-wide shared memory allocation.

### Authentication (`hivemind/utils/auth.py`)

**Class:** `TokenAuthorizerBase` (line 49)

RSA-based request/response signing with nonce replay prevention. Signs requests with private
key, validates with public key. Timestamps checked within ±1 minute.

### TimedStorage (`hivemind/utils/timed_storage.py`)

**Class:** `TimedStorage` (line 50)

Dict-like storage with automatic expiration. Uses min-heap for efficient cleanup. Supports
`maxsize` bound and `freeze()` context for consistent multi-step validation.

### PerformanceEMA (`hivemind/utils/performance_ema.py`)

**Class:** `PerformanceEMA` (line 7)

Exponential moving average throughput tracker with bias correction. Thread-safe via lock.
Used by ProgressTracker to estimate `samples_per_second`.
