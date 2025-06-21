#pragma once

#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace console {

/**
 * @brief Enables virtual terminal processing for ANSI escape codes on Windows.
 * On other platforms, this is a no-op as it's typically enabled by default.
 */
inline void enable_virtual_terminal_processing() {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;

    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) return;

    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
#endif
}

/**
 * @brief Sets the console code page to UTF-8 on Windows.
 * On other platforms, this is a no-op as UTF-8 is usually the default.
 */
inline void setup_utf8_console() {
#ifdef _WIN32
    // 65001 is the code page for UTF-8. "> nul" redirects the output to prevent "Active code page: 65001" from printing.
    system("chcp 65001 > nul");
#endif
}

inline void move_cursor(int row, int col) {
    // ANSI escape codes are 1-based, so we add 1 to the 0-based row and col.
    std::cout << "\x1B[" << row + 1 << ";" << col + 1 << "H";
}

inline void clear_screen() {
    // Clears the screen and moves the cursor to the top-left corner.
    std::cout << "\x1B[2J\x1B[H";
}

} // namespace console