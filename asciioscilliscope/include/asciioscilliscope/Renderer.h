#pragma once
#include <vector>
#include <atomic>
#include <tuple>
#include "PixelFrameBuffer.h"
#include "CharDisplay.h"
#include "CharClassifier.h"

namespace asciioscilliscope {

/**
 * Renderer
 * --------
 * High-level orchestration of image ingestion, phosphor simulation,
 * ASCII classification, and terminal output.
 * Supports causal playback, live input streams, and batch processing.
 *
 * Responsibilities:
 *   - Downsample full-res float images to phosphor grid
 *   - Update PixelFrameBuffer with phosphor data
 *   - Convert buffer diffs into chars via CharClassifier
 *   - Send minimal ANSI sequences to terminal via CharDisplay
 *
 * TODOs:
 *   - Parameterize block size (currently hardcoded 4×4)
 *   - Integrate start_signal_reader for live audio/stream-driven data
 *   - Support time-indexed tensor input for replay
 */
class Renderer {
public:
    /**
     * Constructor
     *
     * @param imgWidth   Full-resolution image width
     * @param imgHeight  Full-resolution image height
     *
     * Initializes:
     *   - phosphor grid dims = img/4
     *   - char grid dims    = img/16
     *   - internal PixelFrameBuffer and CharDisplay
     */
    Renderer(int imgWidth, int imgHeight);

    /**
     * exciteFromImage
     * ---------------
     * Accepts a normalized float image [0.0f,1.0f] in row-major RGB format,
     * downsamples into phosphor-level excitation, and pushes to PFB.
     *
     * @param img  Vector<float> size = imgWidth*imgHeight*3
     */
    void exciteFromImage(const std::vector<float>& img);

    /**
     * start
     * -----
     * Runs the rendering loop until stop() is called or input thread ends.
     * Polls PFB diffs, classifies to chars, and issues ANSI updates.
     */
    void start();

    /**
     * stop
     * ----
     * Requests the render loop to terminate after current iteration.
     */
    void stop();

private:
    int imgW_, imgH_, phosphorW_, phosphorH_, charW_, charH_;
    PixelFrameBuffer<> pfb_;
    CharClassifier classifier_;
    CharDisplay display_;
    std::atomic<bool> running_;

    // Helper to convert PFB diffs into display.set() calls
    void processDiffs(float threshold);

    // Helper to flush ANSI sequences via display.diffAndSwap()
    void flushDisplay();
};

} // namespace asciioscilliscope
