#pragma once

#include <atomic>

// Forward declaration of Screen from ultimate.cpp
class Screen;

/**
 * Launch a thread reading float samples (–1.0 to +1.0) from stdin,
 * map each to a phosphor excitation at a moving index.
 * @param screen           target phosphor screen
 * @param width            screen width
 * @param height           screen height
 * @param running          flag to continue reading
 * @param scanHorizontal   if true, X advances and sample maps to Y; else Y advances and sample maps to X
 * @param fixedIndex       fixed row (if horizontal scan) or column (if vertical scan)
 */
#include <thread>
#include <iostream>
#include <algorithm>

inline void start_signal_reader(
    Screen &screen,
    int width,
    int height,
    std::atomic<bool> &running,
    bool scanHorizontal = true,
    int fixedIndex = 0
) {
    std::thread([&, width, height, scanHorizontal, fixedIndex]() {
        int moving = 0;
        while (running.load()) {
            float sample;
            if (!(std::cin >> sample)) { running.store(false); break; }
            sample = std::clamp(sample, -1.0f, 1.0f);
            uint8_t val = static_cast<uint8_t>((sample + 1.0f) * 127.5f);
            if (scanHorizontal) {
                screen.excite(moving, fixedIndex, val, val, val);
                moving = (moving + 1) % width;
            } else {
                screen.excite(fixedIndex, moving, val, val, val);
                moving = (moving + 1) % height;
            }
        }
    }).detach();
}
