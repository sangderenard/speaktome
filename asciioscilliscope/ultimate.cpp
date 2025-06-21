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
#include <algorithm>
#include "screen.h"
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

// Double-buffered diff manager for RGB pixels
class PixelFrameBuffer {
public:
    PixelFrameBuffer(int rows, int cols)
        : rows(rows), cols(cols), size(rows*cols*3),
          curr(size), prev(size) {}

    void update_render(const std::vector<uint8_t>& data) {
        if (data.size() != size) return;
        std::lock_guard<std::mutex> lk(m);
        curr = data;
    }

    std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> get_diff_and_swap() {
        std::vector<std::tuple<int,int,uint8_t,uint8_t,uint8_t>> diff;
        std::lock_guard<std::mutex> lk(m);
        for (int i=0, idx=0; i<size; i+=3, ++idx) {
            uint8_t r=curr[i], g=curr[i+1], b=curr[i+2];
            uint8_t pr=prev[i], pg=prev[i+1], pb=prev[i+2];
            if (r!=pr || g!=pg || b!=pb) {
                int y = idx / cols, x = idx % cols;
                diff.emplace_back(y,x,r,g,b);
            }
        }
        prev.swap(curr);
        return diff;
    }

private:
    int rows, cols, size;
    std::vector<uint8_t> curr, prev;
    std::mutex m;
};

// Map RGB values to ASCII characters via brightness
class CharClassifier {
public:
    char classify(uint8_t r, uint8_t g, uint8_t b) const {
        static const std::string ramp = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
        float brightness = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        size_t idx = static_cast<size_t>((brightness / 255.f) * (ramp.size() - 1));
        return ramp[idx];
    }
};

struct CharCell {
    char ch{ ' ' };
    uint8_t r{0}, g{0}, b{0};
};

// Double-buffered character grid for terminal output
class CharDisplay {
public:
    CharDisplay(int rows, int cols) : rows(rows), cols(cols),
        curr(rows*cols), next(rows*cols) {}

    void set(int y, int x, char ch, uint8_t r, uint8_t g, uint8_t b) {
        if (x<0||x>=cols||y<0||y>=rows) return;
        next[y*cols + x] = CharCell{ch, r, g, b};
    }

    std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> diff_and_swap() {
        std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>> diff;
        for (int i=0;i<rows*cols;++i) {
            const CharCell &n = next[i];
            CharCell &c = curr[i];
            if (n.ch!=c.ch || n.r!=c.r || n.g!=c.g || n.b!=c.b) {
                diff.emplace_back(i/cols, i%cols, n.ch, n.r, n.g, n.b);
                c = n;
            }
        }
        std::fill(next.begin(), next.end(), CharCell{});
        return diff;
    }

private:
    int rows, cols;
    std::vector<CharCell> curr, next;
};


// Renderer that diffs and draws to console
class Renderer {
public:
    Renderer(int img_w, int img_h)
      : img_w(img_w), img_h(img_h),
        phosphor_w(img_w/4), phosphor_h(img_h/4),
        char_w(img_w/16), char_h(img_h/16),
        fb(char_h, char_w), screen(phosphor_w, phosphor_h),
        display(char_h, char_w),
        img_prev(img_w*img_h*3), ph_prev(phosphor_w*phosphor_h*3),
        running(true) {}

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
            auto ph_curr = screen.render_buffer();
            if (ph_prev.empty()) ph_prev.resize(ph_curr.size());
            for (size_t i=0; i<ph_curr.size(); ++i) {
                volatile bool changed = ph_curr[i] != ph_prev[i];
                (void)changed;
            }
            std::vector<uint8_t> ph_buf = ph_curr;
            ph_prev.swap(ph_curr);
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
            auto diffs = fb.get_diff_and_swap();
            for (auto &t: diffs) {
                int y,x; uint8_t r,g,b;
                std::tie(y,x,r,g,b) = t;
                char ch = classifier.classify(r,g,b);
                display.set(y,x,ch,r,g,b);
            }
            auto char_diffs = display.diff_and_swap();
            draw(char_diffs);
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
public:
    void draw(const std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>>& d) {
        for (auto &t: d) {
            int y,x; char ch; uint8_t r,g,b;
            std::tie(y,x,ch,r,g,b) = t;
            std::cout << "\x1B["<<(y+1)<<";"<<(x+1)<<"H"
                      << "\x1B[38;2;"<<(int)r<<";"<<(int)g<<";"<<(int)b<<"m"<<ch;
        }
        std::cout<<"\x1B[0m"<<std::flush;
    }
    PixelFrameBuffer fb;
    Screen screen;
    CharDisplay display;
    std::vector<uint8_t> img_prev;
    std::vector<uint8_t> ph_prev;
    CharClassifier classifier;
    std::atomic<bool> running;
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
        start_signal_reader(screen, 80, 30, running);
        // draw loop
        Renderer R(80,30); // diff/draw on Screen
        while (running) {
            auto buf = screen.render_buffer();
            R.fb.update_render(buf);
            auto diffs = R.fb.get_diff_and_swap();
            for (auto &t : diffs) {
                int y,x; uint8_t r,g,b; std::tie(y,x,r,g,b)=t;
                char ch = R.classifier.classify(r,g,b);
                R.display.set(y,x,ch,r,g,b);
            }
            auto cd = R.display.diff_and_swap();
            R.draw(cd);
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
