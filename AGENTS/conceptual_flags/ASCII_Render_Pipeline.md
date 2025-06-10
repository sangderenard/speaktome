Here's your **full algorithmic outline** in structured markdown for inclusion in your repo. It breaks down the **triple buffer ASCII rendering pipeline** cleanly, modularly, and with execution flow annotated for any autonomous agent or contributor.

---

# 📜 `docs/ascii_render_pipeline.md`

## 🧭 Overview

This document outlines the design and implementation strategy for a **triple-buffered ASCII render system** enabling **flicker-free, high-performance terminal graphics** using minimal diff redraws.
It supports multithreaded rendering and dynamic terminal updates.

---

## 🧱 Core Concepts

### Buffers:

| Name             | Purpose                         | Ownership     |
| ---------------- | ------------------------------- | ------------- |
| `buffer_render`  | Fresh render from Pillow/Canvas | Render Thread |
| `buffer_next`    | Copied view for update eval     | Main Thread   |
| `buffer_display` | Last terminal frame             | Main Thread   |

### ASCII Render Flow:

1. `render_thread` produces new frame → updates `buffer_render`.
2. `main_loop` reads `buffer_render` → copies to `buffer_next`.
3. `buffer_next` is diffed against `buffer_display` to compute:

   * `(y, x, char)` draw set
4. Only changed positions are redrawn using ANSI cursor movement.
5. `buffer_display` is updated with new frame.

---

## 🧠 Modules

### `framebuffer.py`

```python
class AsciiFrameBuffer:
    def __init__(self, shape: tuple[int, int]):
        ...
    def update_render(self, new_data: np.ndarray):
        ...
    def get_diff_and_promote(self) -> list[tuple[int, int, str]]:
        ...
```

* Manages memory for 3 buffers.
* Thread-safe updates.
* Efficient `np.argwhere` diff engine.

---

### `render_thread.py`

```python
def render_loop(framebuffer: AsciiFrameBuffer, render_fn: Callable[[], np.ndarray], fps: float):
    while running:
        ascii_array = render_fn()
        framebuffer.update_render(ascii_array)
        time.sleep(1 / fps)
```

* Calls `render_fn()` (e.g., clock face, digits).
* Converts Pillow image → ASCII → `np.ndarray`.
* Runs at controlled FPS (default: 10–30).

---

### `draw.py`

```python
def draw_diff(diff_data: list[tuple[int, int, str]]):
    for y, x, char in diff_data:
        print(f"\x1b[{y+1};{x+1}H{char}", end="")
```

* Simple draw loop using ANSI escape codes.
* Efficient; only prints changed characters.

---

### `ascii_digits.py` (modification)

Add:

```python
def render_ascii_to_array(...) -> np.ndarray:
    ...
    return np.array(list_of_char_lists, dtype='<U1')
```

* Returns a 2D array instead of raw `str`.
* Compatible with `AsciiFrameBuffer`.

---

## 🧪 Testing Plan

* ✅ **Single frame test**: verify conversion from PIL → ASCII array
* ✅ **Buffer diff test**: input two arrays, check correct diff output
* ✅ **Render thread test**: run mock animation into buffer, ensure no crash
* ✅ **Live run**: ensure only diff is drawn to terminal

---

## ⚙️ Integration Notes

* Update `clock_demo.py`:

  * Spawn render thread with config
  * Poll framebuffer in UI loop
  * Replace `print()` with `draw_diff(...)`

* Optional: Add toggle key `'d'` to display diff overlay or raw ASCII matrix for diagnostics.

---

## 🧩 Extensibility

* Can support multiple viewports (e.g., analog + digital simultaneously).
* Could add layer compositing, effects, or scrolling in future.
* Hook into audio or input streams for rich interactive displays.

---

## 🧠 Authors

* System architect: **Albert**
* Initial implementation and markup: **GPT-4o / Codex-AI**

---

Let me know if you want me to:

* Drop this into a `docs/` folder as an actual `.md` file.
* Stub out the starter `framebuffer.py` to go with it.
