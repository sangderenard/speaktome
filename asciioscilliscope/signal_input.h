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
void start_signal_reader(
    Screen &screen,
    int width,
    int height,
    std::atomic<bool> &running,
    bool scanHorizontal = true,
    int fixedIndex = 0
);
