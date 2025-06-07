# Standard library imports
import queue as py_queue
import threading
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

# Third-party imports
import torch
# --- END HEADER ---

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from .compressed_beam_tree import CompressedBeamTree  # For type hinting

class BeamRetirementManager:
    """
    Threaded manager for retired beams using hash buckets and queues.
    - add_batch(): push new retirees to a queue (non-blocking, safe).
    - Retirement thread: filters, deduplicates, and makes ready for GPU.
    - get_gpu_batch(): pull up to N beams for GPU batch.
    - Thread-safe, high-throughput.
    """

    def __init__(self, tree: 'CompressedBeamTree', prefix_len=8, tokenizer=None, filter_config=None, queue_size=16384):
        self.tree = tree # Reference to the main beam tree
        self.prefix_len = prefix_len
        self.graph_op = self.tree.operator
        self.tokenizer = tokenizer
        self._bucket: Dict[int, List[int]] = {}  # {prefix_hash: [beam_idx, ...]}
        self._lock = threading.RLock() # RLock for reentrancy if needed
        self.shutdown_flag = threading.Event()
        self.max_len_for_filter = 1024 # Or get from tree config if available

        # Default ASCII filtering setup
        if tokenizer is not None:
            vocab = [tokenizer.decode([i], skip_special_tokens=False) for i in range(tokenizer.vocab_size)]
            self._ascii_good_ids = set(
                i for i, s in enumerate(vocab)
                if all(32 <= ord(c) <= 126 for c in s if len(s) > 0)
            )
            self._strange_token_ids = set(range(tokenizer.vocab_size)) - self._ascii_good_ids
        else:
            self._strange_token_ids = set()
        self._max_strange_total = (filter_config or {}).get("max_total", 3)
        self._max_strange_consecutive = (filter_config or {}).get("max_consecutive", 2)

        self.input_queue = py_queue.Queue(queue_size)
        self.ready_queue = py_queue.Queue(10*queue_size) # Increased ready_queue size
        self.thread = threading.Thread(target=self._retirement_worker, daemon=True)
        self.thread.start()

    def _is_strange(self, beam_idx: int) -> bool:
        # Find the node index for the leaf
        node_idx = self.tree.leaf_node_indices.get(beam_idx)
        if node_idx is None:
            return False

        # Use BeamGraphOperator to collect path
        nodes = [self.tree.nodes[node_idx]]
        data, mask, _ = self.graph_op.pad_nodes_to_relative_tree_depth(
            nodes, pad_token_id=0, device=nodes[0].score_tensor.device, include_unwashed_parents=True
        )
        tokens_flat = data[mask]  # All tokens, flattened, as ints

        if tokens_flat.numel() == 0:
            return False

        # Now build the "strange" lookup as before, but only for the actual tokens
        max_token_val = tokens_flat.max().item()
        # Efficient check for strange tokens
        total_strange_tokens = sum(1 for t_id_tensor in tokens_flat if t_id_tensor.item() in self._strange_token_ids)

        return (total_strange_tokens >= self._max_strange_total)


    def garbage_collect(self, limit_per_bucket=512, total_limit=None):
        with self._lock:
            for h in list(self._bucket.keys()):
                group = self._bucket[h]
                if len(group) > limit_per_bucket:
                    self._bucket[h] = group[-limit_per_bucket:]
            if total_limit is not None:
                all_items_indices = [(h, beam_idx)
                            for h, group in self._bucket.items()
                            for beam_idx in group]
                if len(all_items_indices) > total_limit:
                    # If sorting is needed, scores must be fetched or passed with beam_idx
                    # For simplicity, let's sort by beam_idx (recency, if idx is sequential)
                    all_items_indices.sort(key=lambda x: x[1], reverse=True) # Keep larger (newer) indices
                    kept_indices = all_items_indices[:total_limit]
                    new_bucket = {}
                    for h, beam_idx in kept_indices:
                        new_bucket.setdefault(h, []).append(beam_idx)
                    self._bucket = new_bucket

    def _prefix_hash(self, beam_idx: int) -> int:
        """
        Efficiently computes a hash of the first prefix_len tokens for the given beam_idx,
        by traversing the parent chain in the CompressedBeamTree.

        Assumes self.tree.leaf_node_indices[beam_idx] gives you the leaf node index.
        """
        node_idx = self.tree.leaf_node_indices.get(beam_idx)
        if node_idx is None:
            return 0  # or raise

        # Traverse up the parent chain, collecting tokens (reverse order)
        prefix_tokens = []
        current = node_idx
        while current is not None and len(prefix_tokens) < self.prefix_len:
            node = self.tree.nodes[current]
            # .token_tensor is a single-element tensor
            prefix_tokens.append(int(node.token_tensor.item()))
            current = node.parent_node_idx

        # Tokens collected are in reverse order (leaf→root), so reverse to root→leaf
        prefix_tokens = prefix_tokens[::-1]

        # If shorter than prefix_len, pad (optional: use 0 or -1 or pad_token_id)
        while len(prefix_tokens) < self.prefix_len:
            prefix_tokens.append(0)  # or self.tree.tokenizer.pad_token_id if you want

        return hash(tuple(prefix_tokens))


    def _process_items(self, items):
        # items is now List[beam_idx] or List[(beam_idx, score_at_retirement)]
        # Assuming items is List[beam_idx] for simplicity of "just an index"
        for beam_idx in items:
            if self.tokenizer and self._is_strange(beam_idx): # _is_strange needs tokenizer
                continue
            
            h = self._prefix_hash(beam_idx)
            with self._lock:
                group = self._bucket.setdefault(h, [])
                # Deduplication: beam_idx is unique for a path in the tree.
                # So, if a beam_idx is already in a bucket, it's a duplicate retirement attempt.
                if beam_idx not in group:
                    group.append(beam_idx)
                    try:
                        # Item in ready_queue: (beam_idx, hash_for_bucketing)
                        self.ready_queue.put_nowait((beam_idx, h))
                    except py_queue.Full:
                        print("[WARNING] Ready queue full. Forcing GC.")
                        self.garbage_collect(limit_per_bucket=256)  # Shrink more aggressively
                        try:
                            # Item in ready_queue: (beam_idx, hash_for_bucketing)
                            self.ready_queue.put_nowait((beam_idx, h))
                        except py_queue.Full:
                            print("[FORCE BURN] Still full after GC. Cycling ready queue into bucket.")

                            # 1. Drain the ready queue
                            drained = []
                            while not self.ready_queue.empty():
                                try:
                                    # drained_item is (beam_idx, h)
                                    drained_item = self.ready_queue.get_nowait()
                                    drained.append(drained_item)  # (beam, score, length, h)
                                    self.ready_queue.task_done()
                                except py_queue.Empty:
                                    break

                            # 2. Sort drained beams (Optional: by beam_idx for recency, or random)
                            # Original sort was by score. If not passing score, sort by beam_idx (recency)
                            drained.sort(key=lambda x: x[0], reverse=True) # Keep larger (newer) beam_idx

                            # 3. Decide how many to preserve in the ready queue
                            keep_n = max(int(0.2 * self.ready_queue.maxsize), 1)  # keep top 20%
                            refill = drained[:keep_n]
                            to_bucket = drained[keep_n:]

                            # 4. Refill the ready queue with top-k
                            for beam_idx_refill, h_refill in refill:
                                try:
                                    self.ready_queue.put_nowait((beam_idx_refill, h_refill))
                                except py_queue.Full:
                                    break  # Stop if somehow full

                            # 5. Move the rest into the bucket
                            for beam_idx_bucket, h_bucket in to_bucket:
                                self._bucket.setdefault(h_bucket, []).append(beam_idx_bucket)

                            # 6. Finally, insert the beam that caused the full condition
                            self._bucket.setdefault(h, []).append(beam_idx)
                            print(f"[RECOVERY] Evicted {len(to_bucket)} to bucket, preserved {len(refill)} in queue.")


    def _retirement_worker(self, overflow=None):
        max_retired = 16384
        if overflow is not None:
            # Ensure overflow tensors are on CPU before filtering
            # overflow is now List[beam_idx]
            self._process_items(overflow)
            return

        while not self.shutdown_flag.is_set():
            if len(self) > max_retired:
                self.garbage_collect(max_retired)

            batch = []
            next_item = None

            while True:
                if next_item is None:
                    try:
                        next_item = self.input_queue.get(timeout=0.1)
                    except py_queue.Empty:
                        break
                    if next_item is None:
                        self.shutdown_flag.set()
                        break

                # next_item is beam_idx
                # Batching by length is no longer relevant here as items are just indices

                batch.append(next_item)
                self.input_queue.task_done()
                next_item = None
                if len(batch) >= 256: # Process in chunks
                    break

            if batch:
                self._process_items(batch)


    def add_batch(self, beam_indices: List[int]):
        dropped = []
        for beam_idx in beam_indices:
            try:
                self.input_queue.put_nowait(beam_idx)
            except py_queue.Full:
                dropped.append(beam_idx)
        if dropped:
            self._retirement_worker(overflow=dropped)


    def get_promoted_beam_indices(self, volume_limit: int = None) -> Tuple[List[int], int]:
        # Returns: List of beam_idx, estimated total_volume of these beams
        promoted_beam_indices: List[int] = []
        total_vol, max_len = 0, 0

        while True:
            try:
                # item from ready_queue is (beam_idx, h)
                beam_idx, _ = self.ready_queue.get_nowait()
            except py_queue.Empty:
                break
            
            # Fetch length from the tree for volume calculation
            # We only need the length, so max_len=1 for get_beam_tensors_by_beam_idx is okay if it returns actual length
            # Let's assume get_beam_tensors_by_beam_idx returns full length correctly.
            self.graph_op.move_paths_to_device([self.tree.leaf_node_indices[beam_idx]], device=self.tree.device) # Use tree's device

            _, _, length_scalar = self.tree.get_beam_tensors_by_beam_idx(beam_idx, self.max_len_for_filter)

            next_vol = (len(promoted_beam_indices) + 1) * max(max_len, length_scalar)
            if volume_limit is not None and next_vol > volume_limit:
                self.ready_queue.put((beam_idx, _)) # Put it back with its hash
                break
            
            promoted_beam_indices.append(beam_idx)
            max_len = max(max_len, length_scalar)
            total_vol = next_vol
            self.ready_queue.task_done()
        
        if not promoted_beam_indices:
            return [], 0
            
        return promoted_beam_indices, total_vol

    def remove_used(self, used_idxs):
        """Remove retired beam indices from internal buckets.

        Parameters
        ----------
        used_idxs : Iterable[int]
            Beam indices that were consumed and should be pruned from the
            retirement buckets.
        """

        used = set(used_idxs)
        if not used:
            return

        with self._lock:
            for h in list(self._bucket.keys()):
                remaining = [idx for idx in self._bucket[h] if idx not in used]
                if remaining:
                    self._bucket[h] = remaining
                else:
                    del self._bucket[h]

    def shutdown(self):
        self.shutdown_flag.set()
        self.input_queue.put(None)
        self.thread.join()

    def __len__(self):
        with self._lock:
            return sum(len(group) for group in self._bucket.values())
