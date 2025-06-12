# Backend Buffer Manager Pseudocode

This document sketches two related buffer management designs. Both approaches maintain a persistent worker thread with a constant C interface and use double buffering to keep a Python-accessible copy while backend processes run asynchronously. State updates are queued and applied atomically during a blocking synchronization phase.

---

## 1. OpenGL Backend

```python
class GLBufferManager:
    def __init__(self, shape: tuple[int, ...]):
        self.front = create_gl_buffer(shape)       # buffer visible to Python
        self.back = create_gl_buffer(shape)        # buffer used by GPU workers
        self.update_queue: "Queue[BufferUpdate]" = Queue()
        self.sync_event = threading.Event()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def enqueue(self, update: BufferUpdate) -> None:
        self.update_queue.put(update)

    def synchronize(self) -> None:
        """Block until all queued updates have been processed."""
        self.update_queue.put(None)  # sentinel triggers swap
        self.sync_event.wait()
        self.sync_event.clear()

    def _worker_loop(self) -> None:
        while True:
            update = self.update_queue.get()
            if update is None:
                glFinish()                     # ensure GPU complete
                self.front, self.back = self.back, self.front
                self.sync_event.set()
                continue
            apply_update_to_gl_buffer(self.back, update)
```
```
*Queue items describe data changes (e.g., new tensor values).*
When `synchronize()` is called, a sentinel `None` signals the worker to flush GPU commands, swap buffers, and notify Python via `Event`.

---

## 2. C Backend

```c
struct CBufferManager {
    double* front;
    double* back;
    UpdateQueue queue;      // thread-safe queue of updates
    pthread_t worker;
    pthread_mutex_t sync_mu;
    pthread_cond_t  sync_cv;
};

void* worker_loop(void* arg) {
    struct CBufferManager* mgr = arg;
    for(;;) {
        Update u = queue_pop(&mgr->queue); // blocking
        if (u.type == SYNC) {
            wait_for_all_backend_processes();
            swap(&mgr->front, &mgr->back);
            pthread_mutex_lock(&mgr->sync_mu);
            pthread_cond_signal(&mgr->sync_cv);
            pthread_mutex_unlock(&mgr->sync_mu);
            continue;
        }
        apply_update(mgr->back, &u);
    }
}

void enqueue_update(struct CBufferManager* mgr, Update u) {
    queue_push(&mgr->queue, u);
}

void synchronize(struct CBufferManager* mgr) {
    queue_push(&mgr->queue, make_sync_update());
    pthread_mutex_lock(&mgr->sync_mu);
    pthread_cond_wait(&mgr->sync_cv, &mgr->sync_mu);
    pthread_mutex_unlock(&mgr->sync_mu);
}
```
```
The C version mirrors the OpenGL design but uses `pthread` primitives and a raw memory buffer. Python would interact with this manager via a stable FFI interface while the worker loop processes queued updates and swaps buffers after all backend tasks complete.
```
