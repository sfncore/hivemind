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
8. [Node0 ↔ Hivemind Integration Map](#8-node0--hivemind-integration-map)
9. [Gap Analysis — Training](#9-gap-analysis--training)
10. [Gap Analysis — Inference](#10-gap-analysis--inference)
11. [Ecosystem — Petals, OpenDiLoCo & Prime Intellect](#11-ecosystem--petals-opendiloco--prime-intellect)
12. [Architecture — The RL Flywheel](#12-architecture--the-rl-flywheel)
13. [Proof of Concept — First Iteration](#13-proof-of-concept--first-iteration)

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

---

## 8. Node0 ↔ Hivemind Integration Map

How Node0 (the training rig) maps onto Hivemind's APIs.

### 8.1 Component Mapping

| Node0 component | Hivemind API | Integration point |
|---|---|---|
| Model (LLaMA layers) | `ModuleBackend` wraps each expert/stage | `module_backend.py:45` — `ModuleBackend(module=nn.Module)` |
| Pipeline experts (Head/Body/Tail) | `Server.create()` with expert UIDs | `server.py:88` — each stage registered as separate expert UID |
| PowerSGD wrappers | `PowerSGDGradientAverager` | `power_sgd_averager.py:28` — rank-r gradient compression |
| AutoStepOptimizer | `hivemind.Optimizer` wrapping local optimizer | `optimizer.py:32` — wraps any `torch.optim.Optimizer` |
| Auth client (Pluralis) | `TokenAuthorizerBase` subclass | `utils/auth.py:49` — implement `get_token()` for Pluralis |
| Peer discovery | `DHT` + `get_experts()` | `dht/dht.py:22` + `server/dht_handler.py:81` |
| Config (YAML + run.json) | `Server.create()` kwargs or `hivemind-server` CLI | `hivemind_cli/run_server.py` — supports `--config config.yml` |

### 8.2 Training Loop Contract

Node0 defines *what* to train; Hivemind provides *how* to train it distributedly.

```
Node0 training loop:
  model = define_model()               # Node0: LLaMA layers, pipeline stages
  local_optim = torch.optim.AdamW(...)  # Node0: local optimizer

  opt = hivemind.Optimizer(             # Hivemind: collaborative wrapper
      dht=dht,
      run_id="training_run",
      target_batch_size=10000,          # global batch across all peers
      batch_size_per_step=32,           # local batch per step() call
      optimizer=local_optim,
      use_local_updates=True,           # update locally, average in background
  )
  opt.load_state_from_peers()           # catch up if joining mid-training

  for batch in dataloader:
      loss = model(batch)
      opt.zero_grad()
      loss.backward()
      opt.step()                        # internally: accumulate → track progress
                                         #   → matchmaking → all-reduce → optimizer
                                         #   → state averaging
```

### 8.3 Expert Hosting Pattern

For MoE-style hosting, Node0 registers pipeline stages as experts:

```python
# Each stage is an independent expert with its own UID
server = hivemind.Server.create(
    expert_uids=["model.head", "model.body.0", "model.body.1", "model.tail"],
    expert_cls=CustomExpertClass,     # register_expert_class("custom")
    hidden_dim=4096,
    initial_peers=DHT_PEERS,
    optimizer=None,                    # or optim_cls for server-side training
    start=True,
)
```

### 8.4 Authentication Integration

Node0's Pluralis auth client integrates via `TokenAuthorizerBase` subclass:

```python
class PluralAuthAuthorizer(TokenAuthorizerBase):
    async def get_token(self) -> AccessToken:
        # Call Pluralis API to get/refresh token
        return await pluralis_client.get_access_token()

    def is_token_valid(self, token) -> bool:
        return token is not None and token.expiry > time.time()
```

The authorizer is passed to `DHT(authorizer=...)` and protects all RPC endpoints with
RSA-signed requests and nonce replay prevention (`utils/auth.py:102–164`).

---

## 9. Gap Analysis — Training

### 9.1 Pipeline Parallelism (HIGH)

**Status:** Not supported. Hivemind's MoE grid is *parallel* expert routing, not
sequential pipeline stages.

The grid-based beam search (`beam_search.py:264`) selects k-best experts and executes
them **in parallel** via `_RemoteCallMany` (`moe.py:192`). All selected experts receive
the same input and their outputs are weighted-summed.

Node0's Head → Body → Tail pipeline requires **sequential** data flow. Currently this
must be composed manually by the client:

```python
# Manual sequential composition — no pipeline overlap
head_out = head_expert(input)      # RPC call 1
body_out = body_expert(head_out)   # RPC call 2 (waits for call 1)
tail_out = tail_expert(body_out)   # RPC call 3 (waits for call 2)
```

**What's missing:**
- No sequential dependency tracking between expert UIDs
- No micro-batch pipeline scheduling (GPipe/PipeDream style overlap of fwd/bwd)
- No stage-aware gradient flow optimization
- No fault tolerance for pipeline stages (MoE has `k_min` fallback; pipeline doesn't)

### 9.2 Backward Re-Runs Forward (MEDIUM)

`ModuleBackend.backward()` (line 135) re-executes the forward pass with `enable_grad()`
before calling `torch.autograd.backward()`. This doubles compute per backward step.

- **Fine for:** Small MoE experts (FFN blocks)
- **Expensive for:** Large LLaMA pipeline stages (billions of parameters per stage)
- **Mitigation:** Could implement activation caching if memory allows, but Hivemind's
  stateless design (Runtime doesn't guarantee fwd/bwd ordering) makes this non-trivial

### 9.3 Heterogeneous Hardware (LOW)

`ProgressTracker` (`progress_tracker.py:304`) estimates global ETA from per-peer
`samples_per_second`. The throughput model assumes peers contribute proportionally to
their speed, but doesn't account for:
- Different GPU memory constraints (some peers can't handle large batches)
- Network bandwidth asymmetry beyond what load balancing handles
- Peers with mixed precision capabilities

### 9.4 Optimizer State Size (LOW)

`TrainingStateAverager` averages optimizer statistics (momentum, variance for AdamW)
across peers. For large models, this state can be 2–3x the parameter count. The averaging
round transmits all averaged tensors, which may be bandwidth-prohibitive for very large
models without compression.

PowerSGD (`power_sgd_averager.py`) compresses gradients but not optimizer state. State
averaging uses `state_averaging_compression` which defaults to `NoCompression`.

---

## 10. Gap Analysis — Inference

### 10.1 Inference Is Supported But Implicit

Hivemind **does** support inference:
- `ModuleBackend.__init__` accepts `optimizer=None` (`module_backend.py:45`)
- `Server.create()` / CLI accepts `--optimizer none` (`run_server.py:91–99`)
- `ModuleBackend.forward()` always runs under `torch.no_grad()` (`module_backend.py:100`)
- Backward is a separate RPC — never auto-triggered
- `RemoteExpert.forward()` works for inference out of the box

**However**, there is no explicit inference mode flag. Inference happens implicitly when
the optimizer is None and no backward calls are made.

### 10.2 No Latency-Focused Scheduling (HIGH)

`TaskPool` (`task_pool.py:59`) is throughput-optimized:
- Waits for `min_batch_size` or `timeout` before processing
- FIFO ordering by timestamp
- No request deadlines, SLO targets, or priority classes

For inference, you want minimal latency: process immediately if GPU is idle, or batch
with a tight deadline (e.g., 10ms). The current batching strategy is designed for training
throughput, not serving latency.

### 10.3 No KV Cache / Autoregressive Support (HIGH)

For LLM inference (autoregressive decoding), each token generation step needs access to
the KV cache from previous steps. Hivemind's forward path is **stateless** — each
`ModuleBackend.forward()` call is independent with no state carried between calls.

Supporting autoregressive inference would require:
- Per-session KV cache management on the server
- Session affinity (route sequential tokens to the same server)
- Cache eviction policies
- Cache-aware request routing

### 10.4 No Serving Infrastructure (MEDIUM)

Missing production serving features:

| Feature | Status |
|---|---|
| Health-check / readiness probes | Not implemented |
| Model versioning / A-B routing | Not implemented |
| Request metrics / monitoring | Partial (`StatsReporter` in `runtime.py:161`) |
| Graceful draining | Not implemented (`shutdown()` is immediate) |
| Rate limiting | Not implemented |
| Request authentication | Available via `TokenAuthorizerBase` |

### 10.5 DHT Discovery Latency (MEDIUM)

Expert discovery via beam search queries the DHT per routing decision. For training this
cost is amortized across many forward/backward steps. For inference, each request pays
the discovery latency.

**Mitigation available:** `MoEBeamSearcher` supports DHT result caching (`beam_search.py:77`
— `allow_cache` parameter), and `negative_caching` to avoid re-querying missing experts.
But cache invalidation on expert failure still incurs discovery delay.

### 10.6 No Speculative / Continuous Batching (LOW)

Modern inference servers (vLLM, TGI) use continuous batching to maximize GPU utilization.
Hivemind's `TaskPool` uses fixed batching: accumulate until `min_batch_size` or `timeout`,
then process as one batch. No support for:
- Adding new requests to an in-flight batch
- Preempting long-running requests
- Speculative decoding across pipeline stages

---

## 11. Ecosystem — Petals, OpenDiLoCo & Prime Intellect

Projects built on Hivemind that are directly relevant to the Gas Town architecture.

### 11.1 Petals (Inference + Fine-Tuning)

**Repo:** https://github.com/bigscience-workshop/petals
**Paper:** "Petals: Collaborative Inference and Fine-tuning of Large Models" (2022)

Solves Hivemind's pipeline parallelism gap for inference. Hosts LLM layers as
sequential pipeline stages across a swarm, with automatic routing and fault tolerance.

| Feature | How it works |
|---------|-------------|
| Sequential pipeline | Each node hosts a range of transformer layers; client routes through them in order |
| Inference | Client sends input → layer 0 node → layer 1 node → ... → output |
| Fine-tuning | Frozen base model on swarm; client trains local LoRA adapters against remote activations |
| Fault tolerance | If a node goes down, reroute through alternative nodes hosting same layers |
| KV cache | Maintained server-side per session — solves Hivemind's stateless forward gap |

**Relevance:** Petals provides the inference serving layer. Gas Town agents would hit a
Petals-backed endpoint for completions rather than building pipeline routing from scratch.

### 11.2 OpenDiLoCo / Prime Intellect (Distributed Training + RL)

**Repo:** https://github.com/PrimeIntellect-ai/OpenDiloco
**Paper:** "OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training" (2024)
**Org:** https://www.primeintellect.ai/

Built directly on Hivemind. Implements Google DeepMind's DiLoCo algorithm for
low-communication distributed training.

**How DiLoCo works:**
1. Each worker trains independently for H steps using a local inner optimizer (AdamW)
2. After H steps, workers compute pseudo-gradients (difference from starting point)
3. An outer optimizer (SGD + Nesterov momentum) averages pseudo-gradients across workers
4. Communication reduced by up to 500x vs standard distributed training
5. Uses Hivemind's DHT for peer discovery, libp2p for comms — no master node

**Demonstrated:** Training across two continents, three countries, 90-95% compute utilization.

**Prime Intellect's training runs:**

| Model | Params | Method | Date |
|-------|--------|--------|------|
| INTELLECT-1 | 10B | Decentralized pre-training (OpenDiLoCo on Hivemind) | 2024 |
| INTELLECT-2 | 32B | Decentralized async RL — permissionless swarm | May 2025 |
| INTELLECT-3 | 106B MoE (12B active) | Large-scale RL (centralized) | Nov 2025 |

**INTELLECT-2 infrastructure (most relevant to Gas Town):**

| Component | Purpose |
|-----------|---------|
| PRIME-RL | Async RL training framework — training nodes consume rollouts, produce updated policies |
| SHARDCAST | Broadcasts updated policy weights from training nodes to inference workers |
| TOPLOC | Verifies rollouts from untrusted inference workers |
| Rust orchestrator | Coordinates global pool: hardware checks, heartbeats, task assignment, contribution tracking |

**INTELLECT-2 data loop:**
```
Training nodes produce updated policy
  → SHARDCAST broadcasts weights to inference workers
  → Inference workers generate rollouts (math/coding tasks)
  → TOPLOC verifies rollouts
  → Verified rollouts feed back into RL training
  → Repeat (fully async, no communication overhead)
```

### 11.3 Other Ecosystem Projects

| Project | Relationship | Status |
|---------|-------------|--------|
| PyTorch Lightning | Hivemind strategy plugin for Lightning training loops | Active integration |
| sahajBERT | Collaboratively pretrained Bengali ALBERT-xlarge | Research (2021) |
| CALM | Collaborative Arabic Language Model | Research (2022) |
| Training Transformers Together | NeurIPS 2021 demo — collaborative text-to-image | Research (2021) |
| Hypermind | Fork with blockchain-based incentives | Early stage |
| BitTensor | Similar concept but independent implementation (not built on Hivemind) | Active, separate ecosystem |

---

## 12. Architecture — The RL Flywheel

### 12.1 Core Insight

Gas Town agents do useful work (coding, research, planning) using LLM inference. Every
inference call produces a (prompt, completion) pair. Every bead tracks whether that work
succeeded or failed. This is a natural RL training signal — no synthetic benchmarks, no
labeling teams, no separate data collection phase.

The work IS the training data. The system does useful work and gets smarter as a side effect.

### 12.2 The Loop

```
Gas Town agents request inference
  → Hivemind swarm serves completions (Petals-style pipeline)
  → Agents use completions to do real work
  → Beads track outcomes (completed / failed / reworked)
  → (prompt, completion, outcome) tuples queue as training data
  → Swarm fine-tunes model on accumulated data (OpenDiLoCo-style)
  → Updated adapters hot-swap into inference nodes
  → Better model → better work → better training data → ...
```

### 12.3 Reward Signals

Gas Town produces multiple quality signals without additional labeling effort:

| Signal | Source | Strength | Latency |
|--------|--------|----------|---------|
| Bead completed first attempt | Beads tracker | Strong positive | Minutes to hours |
| Bead required rework | Beads tracker | Weak negative | Minutes to hours |
| Bead failed / abandoned | Beads tracker | Strong negative | Hours |
| Mayor explicit approval | Human review | Strongest positive | Variable |
| Mayor rejection / edit | Human review | Strongest negative | Variable |
| Code compiles + tests pass | CI / verification | Strong positive (verifiable) | Minutes |
| Agent self-correction | Agent retries in same session | Weak negative on first attempt | Immediate |

This maps to standard RL formulations:
- **RLHF / DPO:** Human preference from mayor approvals vs rejections
- **GRPO / outcome-based RL:** Bead success/failure as binary reward
- **Verifiable rewards:** Code compilation and test outcomes (like INTELLECT-2's math verification)

### 12.4 Three Interlocking Flywheels

**Data flywheel:** Useful work → training signal → better model → more useful work

**Compute flywheel:** Contributors join swarm → more capacity → handle more work →
attract more contributors

**Quality flywheel:** Better model → fewer reworks → higher reward signal quality →
more effective training → even better model

### 12.5 Component Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Application | Gas Town (agents, beads, mayor) | Produces work + reward signals |
| API gateway | OpenAI-compatible proxy | Routes agent requests to swarm |
| Inference | Petals on Hivemind | Serves model across distributed nodes |
| Data capture | Logging middleware in API gateway | Records (prompt, completion, metadata) |
| Reward | Beads outcome tracker | Maps bead results to scalar rewards |
| Training | OpenDiLoCo / PRIME-RL on Hivemind | Distributed RL fine-tuning |
| Sync | SHARDCAST-style broadcast | Pushes updated adapters to inference nodes |
| Orchestration | Hivemind DHT + coordinator | Peer discovery, health, task assignment |

### 12.6 Why Fine-Tuning, Not Full Training

Target hardware is 16GB VRAM nodes. With QLoRA:
- Frozen base model (int4 quantized): ~0.5 bytes/param → 7B model fits in ~3.5GB
- LoRA adapters (trainable): 0.1-1% of base params → tiny
- Adapter gradients + optimizer states: ~18 bytes per adapter param only → small
- Remaining VRAM for activations and KV cache

7B-13B models fine-tune comfortably on 16GB. No pipeline parallelism needed when each
node holds the full frozen model. Hivemind's data-parallel `Optimizer` +
`DecentralizedAverager` handles gradient averaging across adapter weights.

For the RL loop specifically: each node generates rollouts (inference) using QLoRA model,
computes rewards from bead outcomes, and trains adapter gradients. Hivemind averages
adapter gradients across the swarm. This is the same pattern as INTELLECT-2 but with
real work instead of synthetic benchmarks.

---

## 13. Proof of Concept — First Iteration

Minimal viable iteration to validate the flywheel on a single machine before distributing.

### 13.1 Scope

**Goal:** Demonstrate that Gas Town agent work produces usable RL training signal, and that
fine-tuning on that signal measurably improves the model at Gas Town tasks.

**Not in scope for PoC:** Multi-node distribution, Petals pipeline, SHARDCAST, NAT
traversal, untrusted compute. Those are scaling concerns — validate the loop first.

### 13.2 Architecture (Single Node)

```
┌─────────────────────────────────────────────────┐
│                  Single Machine                  │
│                                                  │
│  ┌──────────┐    ┌──────────────┐               │
│  │ Gas Town │───▶│  Local vLLM  │               │
│  │  Agents  │◀───│  (7B QLoRA)  │               │
│  └────┬─────┘    └──────────────┘               │
│       │                  ▲                       │
│       │ bead outcomes    │ adapter swap           │
│       ▼                  │                       │
│  ┌──────────┐    ┌──────────────┐               │
│  │  Data    │───▶│   QLoRA      │               │
│  │  Logger  │    │   Trainer    │               │
│  └──────────┘    └──────────────┘               │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 13.3 Components to Build

**Step 1: Local inference server**

Run a 7B model (e.g., Qwen-2.5-7B, Llama-3.1-8B) with QLoRA on a single GPU using
vLLM or llama.cpp. Expose an OpenAI-compatible API endpoint.

```bash
# Example with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --quantization awq \
    --enable-lora \
    --lora-modules gastown-adapter=./adapters/gastown-v0 \
    --port 8000
```

Deliverable: Gas Town agents can hit `http://localhost:8000/v1/chat/completions`
instead of (or alongside) Claude API.

**Step 2: Data capture middleware**

A thin proxy or logging layer between Gas Town agents and the inference server.
Captures every request/response pair with metadata.

```python
# Minimal data logger — captures inference calls
import json, time, uuid
from pathlib import Path

DATA_DIR = Path("./training_data")

def log_inference(prompt: str, completion: str, metadata: dict) -> str:
    """Log a single inference call. Returns log ID."""
    log_id = str(uuid.uuid4())[:8]
    record = {
        "id": log_id,
        "timestamp": time.time(),
        "prompt": prompt,
        "completion": completion,
        "metadata": metadata,  # agent name, bead ID, task type
        "reward": None,        # filled in later by reward mapper
    }
    (DATA_DIR / f"{log_id}.json").write_text(json.dumps(record))
    return log_id
```

Deliverable: Every inference call produces a JSON record with prompt, completion, and
a slot for reward signal.

**Step 3: Reward mapper**

Connects bead outcomes to inference logs. When a bead completes or fails, look up which
inference calls were made during that bead's execution and assign rewards.

```python
# Reward mapper — connects bead outcomes to training data
REWARD_MAP = {
    "completed_first_attempt": 1.0,
    "completed_after_rework": 0.3,
    "failed": -0.5,
    "abandoned": -1.0,
    "mayor_approved": 1.5,
    "mayor_rejected": -1.0,
    "tests_passed": 0.5,   # bonus signal
    "tests_failed": -0.3,  # partial negative
}

def assign_rewards(bead_id: str, outcome: str):
    """Find all inference logs for this bead and assign reward."""
    reward = REWARD_MAP.get(outcome, 0.0)
    for log_file in DATA_DIR.glob("*.json"):
        record = json.loads(log_file.read_text())
        if record["metadata"].get("bead_id") == bead_id:
            record["reward"] = reward
            log_file.write_text(json.dumps(record))
```

Deliverable: Inference logs get reward values based on bead outcomes.

**Step 4: QLoRA fine-tuning script**

Periodically trains LoRA adapters on accumulated (prompt, completion, reward) data.
For the PoC, this can be a simple offline script run manually or on a cron.

```python
# Simplified — actual implementation would use trl + peft
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# Load accumulated training data
dataset = load_reward_dataset("./training_data/")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)

# GRPO: Group Relative Policy Optimization
# Uses reward signal directly, no separate reward model needed
training_config = GRPOConfig(
    output_dir="./adapters/gastown-v1",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
)

trainer = GRPOTrainer(
    model=base_model,
    config=training_config,
    peft_config=lora_config,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model("./adapters/gastown-v1")
```

Deliverable: A new adapter checkpoint trained on Gas Town's own data.

**Step 5: Adapter hot-swap**

Load the new adapter into the running inference server without restart.

```bash
# vLLM supports dynamic LoRA loading
curl -X POST http://localhost:8000/v1/load_lora \
    -d '{"lora_name": "gastown-adapter", "lora_path": "./adapters/gastown-v1"}'
```

Or for llama.cpp / Ollama, swap the adapter file and reload.

Deliverable: Inference server now uses the fine-tuned adapter. Loop complete.

### 13.4 Validation Metrics

How to know the flywheel is working:

| Metric | Measure | Target for PoC |
|--------|---------|----------------|
| Bead success rate | % beads completed first attempt, before vs after fine-tune | Any measurable improvement |
| Rework rate | % beads requiring rework, before vs after | Decrease |
| Reward trend | Average reward per batch over time | Upward trend |
| Completion quality | Human eval (mayor) of random samples | Subjective improvement |
| Task-specific perf | Code compilation rate, test pass rate | Improvement on code tasks |

**Baseline:** Run Gas Town on the stock model for N beads. Record metrics.
**Treatment:** Run Gas Town on the fine-tuned model for N beads. Compare.

### 13.5 Minimum Hardware

| Component | Requirement |
|-----------|------------|
| GPU | 1x 16GB+ (RTX 4060 Ti 16GB, RTX 3090, A4000, etc.) |
| RAM | 32GB+ (optimizer offload, data loading) |
| Storage | 50GB+ (model weights, training data, adapter checkpoints) |
| Software | vLLM or llama.cpp, PyTorch, PEFT, TRL |

### 13.6 PoC → Distributed (What Comes After)

Once the single-node flywheel is validated:

| Step | What changes | Technology |
|------|-------------|-----------|
| 1. Multi-node inference | Model hosted across nodes instead of single GPU | Petals on Hivemind |
| 2. Distributed fine-tuning | Multiple nodes train adapters, average gradients | OpenDiLoCo / Hivemind Optimizer |
| 3. Async RL loop | Training and inference overlap continuously | PRIME-RL pattern |
| 4. Adapter broadcast | Push updated adapters to all inference nodes | SHARDCAST-style on Hivemind DHT |
| 5. Permissionless compute | Anyone can contribute GPU to the swarm | Hivemind P2P + auth |
| 6. Untrusted verification | Verify rollouts from unknown contributors | TOPLOC-style verification |

Each step is independently valuable. Step 1 alone gives decentralized inference.
Steps 1+2 give the distributed flywheel. Steps 3-6 are scaling and hardening.

### 13.7 Open Questions for PoC

1. **Base model selection:** Qwen-2.5-7B-Instruct vs Llama-3.1-8B-Instruct vs
   DeepSeek-R1-Distill-7B? Needs benchmarking on Gas Town task types (code, planning,
   research). Code-specialized models may have better starting performance.

2. **RL algorithm:** GRPO (no reward model needed, uses group relative ranking) vs
   DPO (needs preference pairs — could construct from mayor approve/reject) vs
   simple SFT on successful completions only (lowest complexity, may be sufficient
   for PoC).

3. **Training frequency:** How many bead completions before a fine-tuning round?
   Too few = noisy gradients. Too many = slow feedback loop. Start with batches
   of ~100 completed beads and adjust.

4. **Evaluation gate:** Should updated adapters be eval'd before deployment, or
   deploy immediately and monitor? For PoC, a simple perplexity check + manual
   spot-check is probably sufficient. Production needs automated eval.

5. **Data mix:** Pure Gas Town data vs Gas Town data mixed with general instruction
   data? Pure risks catastrophic forgetting of general capabilities. A 50/50 mix
   with a general dataset (like OpenHermes) is safer.

6. **Which Gas Town tasks to start with:** Code-heavy tasks have the clearest
   reward signal (compiles/doesn't compile, tests pass/fail). Research and planning
   tasks have noisier rewards. Start with code tasks for clearest signal.
