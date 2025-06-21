#pragma once
#include <vector>
#include <tuple>
#include <cstdint>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <deque>
#include <mutex>

namespace asciioscilliscope {

struct CharCell {
    char ch;
    uint8_t r, g, b;
};

/**
 * CharDisplay
 * -----------
 * Manages time-sliced character frames with a classic double-buffer flipper.
 * Queues incoming slices, computes per-cell diffs, and alternates display buffers
 * for low-latency updates. Supports full-print vs. diff-only modes.
 *
 * Responsibilities:
 *   - Accept batched/delayed char frames with timestamps
 *   - Maintain a deque of (timestamp, slice) entries
 *   - On getNextDiff(): pop one slice, diff against current buffer,
 *     swap active buffer, and return raw diffs
 *   - Provide full buffer for full-print mode
 *
 * Thread Safety:
 *   - All public methods lock a mutex for multi-threaded producers/consumers
 */
class CharDisplay {
public:
    /**
     * Constructor
     *
     * @param rows           Number of text rows
     * @param cols           Number of text columns
     * @param fullPrintMode  If true, full buffer is returned instead of diffs
     */
    CharDisplay(int rows, int cols, bool fullPrintMode = false);

    /**
     * stageSlice
     * -----------
     * Enqueue a complete character frame for scheduled display.
     *
     * @param slice      Eigen::Tensor<char,2> [rows,cols] of ASCII codes
     * @param timestamp  Desired display time or sequence index
     */
    void stageSlice(const Eigen::Tensor<char,2>& slice, double timestamp);

    /**
     * getNextDiff
     * -----------
     * Pop the next scheduled slice, compute diffs against the active buffer,
     * swap buffers, and return per-cell diff entries.
     *
     * @return Vector of (row, col, char, r, g, b) for updated cells
     */
    std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>>
    getNextDiff();
    
    /**
     * getFullBuffer
     * -------------
     * Returns the full current buffer (rows × cols)
     * for full-print mode. Thread-safe.
     */
    const Eigen::Tensor<char,2>& getFullBuffer() const;

private:
    int rows_, cols_;
    bool fullPrintMode_;
    std::deque<std::pair<double, Eigen::Tensor<char,2>>> sliceQueue_;
    Eigen::Tensor<char,2> buffers_[2];
    int activeBuffer_ = 0;
    mutable std::mutex mutex_;

    // Internal helpers: diff computation, buffer rotation
    // TODO: implement efficient Eigen-based diff and mutex locking
};

} // namespace asciioscilliscope
