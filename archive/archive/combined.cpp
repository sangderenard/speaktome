#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <algorithm>

// ########## STUB: PixelFrameBuffer ##########
// PURPOSE: Manage triple frame buffers for diff based terminal drawing in C++.
// EXPECTED BEHAVIOR: Maintain render, next, and display buffers of RGB pixels.
// INPUTS: RGB pixel data via update_render().
// OUTPUTS: Changed pixel tuples (y,x,r,g,b) from get_diff_and_promote().
// KEY ASSUMPTIONS/DEPENDENCIES: Uses a flat byte array in row-major order with
// stride of 3 for color channels.
// TODO:
//   - [ ] Implement resize support for dynamic resolution changes
//   - [ ] Expose diff threshold configuration
// NOTES: This is a simplified port of timesync.frame_buffer.PixelFrameBuffer.
// ###########################################################################
class PixelFrameBuffer {
public:
    PixelFrameBuffer(int rows, int cols)
        : m_rows(rows), m_cols(cols), m_size(rows * cols * 3),
          buffer_render(m_size, 0), buffer_next(m_size, 0),
          buffer_display(m_size, 0) {}

    void update_render(const std::vector<uint8_t> &data) {
        if (data.size() != buffer_render.size()) return; // naive check
        buffer_render = data;
    }

    std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> get_diff_and_promote() {
        std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> changed;
        for (int i = 0; i < m_size; i+=3) {
            if (buffer_render[i] != buffer_display[i] ||
                buffer_render[i+1] != buffer_display[i+1] ||
                buffer_render[i+2] != buffer_display[i+2]) {
                int idx = i / 3;
                changed.emplace_back(idx / m_cols, idx % m_cols,
                    buffer_render[i], buffer_render[i+1], buffer_render[i+2]);
            }
            buffer_next[i] = buffer_render[i];
            buffer_next[i+1] = buffer_render[i+1];
            buffer_next[i+2] = buffer_render[i+2];
        }
        buffer_display.swap(buffer_next);
        return changed;
    }

private:
    int m_rows;
    int m_cols;
    int m_size;
    std::vector<uint8_t> buffer_render;
    std::vector<uint8_t> buffer_next;
    std::vector<uint8_t> buffer_display;
};

// ########## STUB: Phosphor ##########
// PURPOSE: Represent a pixel with decaying intensity.
// EXPECTED BEHAVIOR: Intensity fades exponentially until excited.
// INPUTS: excite(level) sets intensity.
// OUTPUTS: get_intensity() returns decayed intensity.
// KEY ASSUMPTIONS/DEPENDENCIES: None.
// TODO: Allow colour per phosphor rather than greyscale.
// ###########################################################################
class Phosphor {
public:
    Phosphor() : intensity(0.0f), lastActivation(std::chrono::steady_clock::now()) {}

    void excite(float level) {
        intensity = std::min<float>(intensity + level, MAX_INTENSITY);
        lastActivation = std::chrono::steady_clock::now();
    }

    float get_intensity() const {
        using namespace std::chrono;
        auto now = steady_clock::now();
        auto ms = duration_cast<milliseconds>(now - lastActivation).count();
        float decay = std::exp(-ms * 0.005f);
        return intensity * decay;
    }

private:
    static constexpr float MAX_INTENSITY = 64.0f;
    float intensity;
    std::chrono::steady_clock::time_point lastActivation;
};

class Screen {
public:
    Screen(int w, int h)
        : width(w), height(h), grid(h, std::vector<Phosphor>(w)) {}

    void excite(int x, int y, float level) {
        if (x>=0 && x<width && y>=0 && y<height) {
            grid[y][x].excite(level);
        }
    }

    std::vector<uint8_t> render_to_buffer() const {
        std::vector<uint8_t> buf(width * height * 3, 0);
        for (int y=0; y<height; ++y) {
            for (int x=0; x<width; ++x) {
                float v = grid[y][x].get_intensity();
                uint8_t c = static_cast<uint8_t>(std::clamp(v, 0.f, 64.f) * 4);
                size_t idx = (y*width + x) * 3;
                buf[idx] = c;
                buf[idx+1] = c;
                buf[idx+2] = c;
            }
        }
        return buf;
    }

private:
    int width;
    int height;
    std::vector<std::vector<Phosphor>> grid;
};

class Renderer {
public:
    Renderer(int w, int h) : fb(h, w), screen(w, h) {}

    void run() {
        using namespace std::chrono_literals;
        std::atomic<bool> running{true};

        std::thread input([&](){
            while(running) {
                float val; if(!(std::cin>>val)) { running=false; break; }
                int y = static_cast<int>((val+1.f)/2.f * (fb_rows()-1));
                screen.excite(cursor_x, y, 8.f);
                cursor_x = (cursor_x + 1) % fb_cols();
            }
        });

        while(running) {
            auto buf = screen.render_to_buffer();
            fb.update_render(buf);
            auto changed = fb.get_diff_and_promote();
            draw_diff(changed);
            std::this_thread::sleep_for(16ms);
        }
        input.join();
    }

private:
    int fb_rows() const { return screen_height; }
    int fb_cols() const { return screen_width; }

    void draw_diff(const std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> &pix) {
        for (const auto &p: pix) {
            int y,x; uint8_t r,g,b; std::tie(y,x,r,g,b)=p;
            std::cout << "\x1B["<<y+1<<";"<<x+1<<"H"
                      << "\x1B[48;2;"<<(int)r<<";"<<(int)g<<";"<<(int)b<<"m"
                      << ' ';
        }
        std::cout << "\x1B[0m" << std::flush;
    }

    int screen_width = 80;
    int screen_height = 40;
    int cursor_x = 0;
    PixelFrameBuffer fb;
    Screen screen;
};

#ifdef _WIN32
#include <windows.h>
inline void enable_virtual_terminal_processing() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;

    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) return;

    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
}

inline void setup_utf8_console() {
    system("chcp 65001 > nul");
}
#endif

int main(){
#ifdef _WIN32
    enable_virtual_terminal_processing();
    setup_utf8_console();
#endif

    Renderer r(80,40);
    r.run();
    return 0;
}
