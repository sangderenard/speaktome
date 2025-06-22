#pragma once
#include <vector>
#include <array>
#include <chrono>
#include <cstdint>
#include <algorithm>

using Clock = std::chrono::steady_clock;

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
