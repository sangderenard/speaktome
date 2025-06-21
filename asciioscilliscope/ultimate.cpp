// ultimate.cpp
// Ultimate ASCII color renderer with triple buffering and phosphor decay
#define STB_IMAGE_IMPLEMENTATION
#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <cstring>
#include "stb_image.h"
#include "signal_input.h"  // Add signal reader header

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

using Clock = std::chrono::steady_clock;
using ms = std::chrono::milliseconds;

// Triple-buffered diff manager for RGB pixels
class PixelFrameBuffer {
public:
    PixelFrameBuffer(int rows, int cols)
        : rows(rows), cols(cols), size(rows*cols*3),
          buf_render(size), buf_next(size), buf_disp(size) {}

    void update_render(const std::vector<uint8_t>& data) {
        if (data.size() != size) return;
        std::lock_guard<std::mutex> lk(m);
        buf_render = data;
    }

    std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> get_diff_and_promote() {
        std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> diff;
        std::lock_guard<std::mutex> lk(m);
        for (int i=0, idx=0; i<size; i+=3, ++idx) {
            uint8_t r=buf_render[i], g=buf_render[i+1], b=buf_render[i+2];
            uint8_t pr=buf_disp[i], pg=buf_disp[i+1], pb=buf_disp[i+2];
            if (r!=pr || g!=pg || b!=pb) {
                int y = idx / cols, x = idx % cols;
                diff.emplace_back(y,x,r,g,b);
            }
            buf_next[i]=r; buf_next[i+1]=g; buf_next[i+2]=b;
        }
        buf_disp.swap(buf_next);
        return diff;
    }

private:
    int rows, cols, size;
    std::vector<uint8_t> buf_render, buf_next, buf_disp;
    std::mutex m;
};

// Color phosphor with independent decay per channel
class ColorPhosphor {
public:
    ColorPhosphor() {
        intensity.fill(0.0f);
        auto now = Clock::now();
        last.fill(now);
    }
    void excite(uint8_t r, uint8_t g, uint8_t b) {
        excite_chan(0, r);
        excite_chan(1, g);
        excite_chan(2, b);
    }
    std::array<uint8_t,3> value() const {
        std::array<uint8_t,3> out;
        auto now = Clock::now();
        for (int c=0; c<3; ++c) {
            float dt = std::chrono::duration<float>(now - last[c]).count();
            float decay = std::exp(-dt * decay_rate);
            float v = intensity[c] * decay;
            out[c] = static_cast<uint8_t>(std::clamp(v,0.0f,255.0f));
        }
        return out;
    }
private:
    void excite_chan(int c, uint8_t level) {
        intensity[c] = std::min(intensity[c] + level, 255.0f);
        last[c] = Clock::now();
    }
    static constexpr float decay_rate = 0.5f; // per second
    std::array<float,3> intensity;
    std::array<Clock::time_point,3> last;
};

// Screen of phosphors
class Screen {
public:
    Screen(int w, int h): width(w), height(h), grid(w*h) {}
    void excite(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x<0||x>=width||y<0||y>=height) return;
        grid[y*width + x].excite(r,g,b);
    }
    std::vector<uint8_t> render_buffer() const {
        std::vector<uint8_t> out(width*height*3);
        for (int y=0,i=0; y<height; ++y) {
            for (int x=0; x<width; ++x, i+=3) {
                auto v = grid[y*width + x].value();
                out[i]=v[0]; out[i+1]=v[1]; out[i+2]=v[2];
            }
        }
        return out;
    }
private:
    int width, height;
    std::vector<ColorPhosphor> grid;
};

// Renderer that diffs and draws to console
class Renderer {
public:
    Renderer(int img_w, int img_h)
      : img_w(img_w), img_h(img_h),
        phosphor_w(img_w/4), phosphor_h(img_h/4),
        char_w(img_w/16), char_h(img_h/16),
        fb(char_h, char_w), screen(phosphor_w, phosphor_h), running(true) {}

    void excite_from_image(const std::vector<uint8_t>& img) {
        // Aggregate image pixels into phosphor grid (4x4 blocks)
        for (int py = 0; py < phosphor_h; ++py) {
            for (int px = 0; px < phosphor_w; ++px) {
                int sum_r=0, sum_g=0, sum_b=0;
                for (int by=0; by<4; ++by) for (int bx=0; bx<4; ++bx) {
                    int ix = px*4 + bx;
                    int iy = py*4 + by;
                    int idx = (iy*img_w + ix)*3;
                    sum_r += img[idx]; sum_g += img[idx+1]; sum_b += img[idx+2];
                }
                // average per block
                uint8_t ar = sum_r/16, ag = sum_g/16, ab = sum_b/16;
                screen.excite(px, py, ar, ag, ab);
            }
        }
    }
    void start() {
        // input thread to exit on Enter
        std::thread input([&]{ std::cin.get(); running=false; });
        // hide cursor
        std::cout<<"\x1B[?25l";
        while(running) {
            auto ph_buf = screen.render_buffer();
            // Downsample phosphor buffer to char grid (4x4 blocks)
            std::vector<uint8_t> char_buf(char_w*char_h*3);
            for (int cy=0; cy<char_h; ++cy) {
                for (int cx=0; cx<char_w; ++cx) {
                    int sr=0, sg=0, sb=0;
                    for (int by=0; by<4; ++by) for (int bx=0; bx<4; ++bx) {
                        int sx = cx*4 + bx;
                        int sy = cy*4 + by;
                        int pidx = (sy*phosphor_w + sx)*3;
                        sr += ph_buf[pidx]; sg += ph_buf[pidx+1]; sb += ph_buf[pidx+2];
                    }
                    int cidx = (cy*char_w + cx)*3;
                    char_buf[cidx]   = sr/16;
                    char_buf[cidx+1] = sg/16;
                    char_buf[cidx+2] = sb/16;
                }
            }
            fb.update_render(char_buf);
            auto diffs = fb.get_diff_and_promote();
            draw(diffs);
            std::this_thread::sleep_for(ms(33));
        }
        // show cursor
        std::cout<<"\x1B[?25h";
        input.join();
    }
private:
    int img_w, img_h;
    int phosphor_w, phosphor_h;
    int char_w, char_h;
    int fb_rows() const { return char_h; }
    int fb_cols() const { return char_w; }
    void draw(const std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>>& d) {
        for (auto &t: d) {
            int y,x; uint8_t r,g,b;
            std::tie(y,x,r,g,b) = t;
            std::cout << "\x1B["<<(y+1)<<";"<<(x+1)<<"H"
                      << "\x1B[48;2;"<<(int)r<<";"<<(int)g<<";"<<(int)b<<"m " ;
        }
        std::cout<<"\x1B[0m"<<std::flush;
    }
    PixelFrameBuffer fb;
    Screen screen;
    std::atomic<bool> running;
    int screen_width, screen_height;
};

int main(int argc, char** argv) {
#ifdef _WIN32
    enable_virtual_terminal_processing();
    setup_utf8_console();
#endif
    if (argc<2) {
        std::cerr<<"Usage: "<<argv[0]<<" <image.png> or `signal` to read stdin floats\n";
        return 1;
    }

    std::atomic<bool> running{true};
    if (std::string(argv[1]) == "signal") {
        // Oscilloscope mode: no image load
        Screen screen(80, 30);
        start_signal_reader(screen, 80, running);
        // draw loop
        Renderer R(80,30); // reinterpret Renderer as diff/draw on Screen?
        while (running) {
            auto buf = screen.render_buffer();
            R.fb.update_render(buf); // assume fb/public
            auto diffs = R.fb.get_diff_and_promote();
            R.draw(diffs);
            std::this_thread::sleep_for(ms(33));
        }
    } else {
        int w,h,ch;
        unsigned char* data = stbi_load(argv[1], &w,&h,&ch,3);
        if (!data) { std::cerr<<"Failed to load image\n"; return 1; }
        std::vector<uint8_t> img(data, data+ w*h*3);
        stbi_image_free(data);

        Renderer R(w,h);
        R.excite_from_image(img);
        R.start();
    }
    return 0;
}
