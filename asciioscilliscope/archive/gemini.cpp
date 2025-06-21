#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory> 
#include "console_utils.h"
#include <algorithm> 
#include <cmath>



#include <iostream>
#include <sstream>
#include <map>
#include <string>
#include <chrono>
#include <mutex>

std::string repeatUTF8String(const std::string& str, int n) {
    std::string result;
    for (int i = 0; i < n; ++i) {
        result += str;
    }
    return result;
}
static std::string addBorder(const std::string& text) {
    std::istringstream iss(text);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(iss, line)) {
        lines.push_back(line);
    }

    int maxLineLength = 0;
    for (const std::string& l : lines) {
        maxLineLength = std::max<int>(maxLineLength, (int)l.length());
    }
    // border characters
    std::string topLeft = "\u250C";     
    std::string topRight = "\u2510";    
    std::string bottomLeft = "\u2514";  
    std::string bottomRight = "\u2518"; 
    std::string horizontal = "\u2500";  
    std::string vertical = "\u2502";    
    std::ostringstream result;
    
    result << topLeft << repeatUTF8String(horizontal, maxLineLength + 2) << topRight << "\n";

    for (const std::string& l : lines) {
        result << vertical << " " << l << std::string(maxLineLength - l.length() + 1, ' ') << vertical << "\n";
    }
    result << bottomLeft << repeatUTF8String(horizontal, maxLineLength + 2) << bottomRight << "\n";

    return result.str();
}
class StatusBlock {
public:
    void addStatus(const std::string& key, const std::string& value) {
        const std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(m);
        statuses[key] = std::make_pair(value, now);
    }

    std::string getStatusString(const std::string& delineator, bool border) {
        const std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(m);

        std::ostringstream oss;
        for (const std::pair<const std::string, std::pair<std::string, std::chrono::steady_clock::time_point>>& item : statuses) {
            const std::string& key = item.first;
            const std::string& value = item.second.first;
            std::chrono::steady_clock::time_point timestamp = item.second.second;
            float elapsedSeconds = std::chrono::duration<float>(now - timestamp).count();

            if (elapsedSeconds < maxAgeSeconds) {
                oss << key << ": " << value << delineator;
            }
        }
        
        if (border) {
            return addBorder(oss.str());
        }
        else {
            return oss.str();
        }
    }

private:
    
    std::map<std::string, std::pair<std::string, std::chrono::time_point<std::chrono::steady_clock>>> statuses;
    std::mutex m;
    static constexpr float maxAgeSeconds = 5.0f; // Adjust as needed
};


StatusBlock globalStatusBlock;






const int console_width = 80;
const int console_height = 40;
//const int max_intensity = 9;
const int max_intensity = 64;
const int decay_rate = 1;

float normalizeSignal(float rawSignal) {
    // Placeholder normalization function
    return rawSignal;
}

class Phosphor {
private:
    std::atomic<float> intensity{ 0.0f };
    std::chrono::steady_clock::time_point lastActivation;
    static constexpr float decayConstant = -0.01f; // this controls the phosphor fade

public:
    Phosphor() : lastActivation(std::chrono::steady_clock::now()) {}

    void excite(float level) {
        // Update the current intensity based on time elapsed since last activation before excitation
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastActivation).count();
        float decayedIntensity = calculateDecayedIntensity(elapsed);

        // Excite phosphor with decay considered
        intensity.store(std::min<float>(decayedIntensity + level, max_intensity));
        lastActivation = now; // Update last activation time
    }

    float getIntensity() const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastActivation).count();
        return calculateDecayedIntensity(elapsed);
    }

private:
    float calculateDecayedIntensity(long long elapsedMilliseconds) const {
        float decayScale = 0.995f; // Adjust decayScale for stronger/weaker nonlinear decay
        float currentIntensity = intensity.load();
        return currentIntensity * std::pow(decayScale, elapsedMilliseconds);
    }
};


//float calculateDecayedIntensity(long long elapsedMilliseconds) const {
//float currentIntensity = intensity.load();
//float stretchedExp = std::exp(-std::pow((elapsedMilliseconds / tau), beta)); // tau and beta are parameters to tune
//return currentIntensity * stretchedExp;
//}
 

// the phosphor sites accessible to the beam
std::vector<std::vector<std::unique_ptr<Phosphor>>> grid;
class Screen {
private:
    std::vector<std::vector<std::unique_ptr<Phosphor>>>& grid;

public:
    Screen(std::vector<std::vector<std::unique_ptr<Phosphor>>>& gridRef) : grid(gridRef) {}

    std::vector<std::string> renderToStringBuffer() {
        std::vector<std::string> stringBuffer(console_height, std::string(console_width, ' '));
        //const std::string intensityChars = " .:-=+*#%@";
        const std::string intensityChars = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";


        for (size_t row = 0; row < grid.size(); ++row) {
            for (size_t col = 0; col < grid[row].size(); ++col) {
                float intensity = grid[row][col]->getIntensity();
                size_t charIndex = static_cast<size_t>((intensity / max_intensity) * (intensityChars.size() - 1));
                stringBuffer[row][col] = intensityChars[charIndex];
            }
        }
        return stringBuffer;
    }
};

void initializeGrid() {
    grid.resize(console_height);
    for (auto& row : grid) {
        row.reserve(console_width);
        for (int col = 0; col < console_width; ++col) {
            row.push_back(std::make_unique<Phosphor>());
        }
    }
}

class InputCache {
public:
    //respect input time delays
    struct Signal {
        float value;
        std::chrono::steady_clock::time_point timestamp;
    };

private:
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<Signal> signals;
    bool ready = false;
    bool isBinaryMode = false;
    
public:
    bool tryGetLatestSignal(Signal& outSignal) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!signals.empty()) {
            outSignal = signals.front();
            signals.pop();
            if (signals.empty()) ready = false;
            return true;
        }
        return false;
    }
public:
    void setBinaryMode(bool enable) {
        isBinaryMode = enable;
    }
    void addSignal(float value) {
        std::lock_guard<std::mutex> lock(mtx);
        signals.push({ value, std::chrono::steady_clock::now() });
        ready = true;
        cv.notify_one();
    }

    Signal getSignal() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return ready; });
        Signal signal = signals.front();
        signals.pop();
        if (signals.empty()) ready = false;
        return signal;
    }
    void processInput(std::atomic<bool>& running) {
        while (running.load()) {
            if (isBinaryMode) {
                double value;
                std::cin.read(reinterpret_cast<char*>(&value), sizeof(double));
                if (std::cin) {  // Check for successful read
                    //std::cout << value;
                    if (value > 1) {
                        exit(1);
                    }
                    addSignal(value);
                }
            }
            else { // Float mode
                float value;
                if (std::cin >> value) {
                    addSignal(value);
                }
            }

            //std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

InputCache inputCache;

void inputHandler(std::atomic<bool>& running) {
    while (running.load()) {
        float value;
        if (std::cin >> value) {
            inputCache.addSignal(value);
        }
    }
}

void beamScan(std::atomic<bool>& running) {
    float peakValues[console_width] = { 0 };
    InputCache::Signal latestSignal;
    float currentPosition = 0.0f;
    float previousPosition = 0.0f;
    float currentSignalValue = 0;
    float previousSignalValue = 0;
    float max_peak = 0;
    float min_scale_factor = 1;
    float effectiveScalingFactor = 1;
    const int baseTickCount = 10; // Starting number of tick marks
    float baseTickSpacing = console_height / (float)baseTickCount; // Initial spacing based on console height

    while (running.load()) {


        if (inputCache.tryGetLatestSignal(latestSignal)) {
            size_t currentColIdx = static_cast<size_t>(currentPosition);
            //Erase the previous peak at this point with the current value
            peakValues[currentColIdx] = abs(latestSignal.value);
            float decayFactor = 0.99999f; // Controls short-term decay
            float scaleFactor = 1 / max_peak;
            max_peak = std::max<float>(max_peak * decayFactor, *std::max_element(peakValues, peakValues + console_width));
            min_scale_factor = std::min<float>(min_scale_factor, 1.0f / max_peak);

            float influenceFactor = 0.2f; // Adjust between 0.0 and 1.0 for desired influence
            effectiveScalingFactor = (1.0f - influenceFactor) * scaleFactor + influenceFactor * min_scale_factor;
            currentSignalValue = latestSignal.value * effectiveScalingFactor;  // Use the influenced scaling factor


            globalStatusBlock.addStatus("Signal Value", std::to_string(latestSignal.value));
            globalStatusBlock.addStatus("Scale Value", std::to_string(1 / max_peak));
            globalStatusBlock.addStatus("Display Value", std::to_string(currentSignalValue));
        } else {
            currentSignalValue = previousSignalValue;
        }
        previousPosition = currentPosition;
        previousSignalValue = currentSignalValue;
        size_t currentRowIdx = static_cast<size_t>((currentSignalValue + 1) / 2 * (console_height - 1));
        size_t previousRowIdx = static_cast<size_t>((previousSignalValue + 1) / 2 * (console_height - 1));
        bool beamwasoff = false;
        // Ensure positions are within bounds
        if (currentPosition >= console_width) {
            currentPosition -= console_width;
            beamwasoff = true;
        }
        if (currentRowIdx >= grid.size()) {
            currentRowIdx = grid.size() - 1;
        }
        // Dynamically adjust tick count or spacing based on the effective scaling factor
        float adjustedTickSpacing = baseTickSpacing * effectiveScalingFactor; // Example adjustment

        // Calculate the dynamic number of ticks based on adjusted spacing and console height
        int dynamicTickCount = std::min<int>((int)(console_height / adjustedTickSpacing), console_height);

        // Clear previous tick marks if necessary

        // Draw new tick marks based on dynamic count and spacing
        for (int i = 0; i < dynamicTickCount; ++i) {
            size_t tickRowIdx = i * (console_height / dynamicTickCount);
            // Ensure the tickRowIdx is within bounds and draw the tick mark
            if (tickRowIdx < console_height) {
                grid[tickRowIdx][0]->excite(max_intensity * 0.5); // Example: mark on left
                grid[tickRowIdx][console_width - 1]->excite(max_intensity * 0.5); // Example: mark on right
            }
        }

     

        // Interpolation and excitement of the path
        float distance = std::sqrt(std::pow(currentPosition - previousPosition, 2) + std::pow(currentRowIdx - previousRowIdx, 2));
        int steps = static_cast<int>(distance * 1.0f); // Determine the number of steps based on distance

        if (beamwasoff) {
            grid[currentRowIdx][currentPosition]->excite(max_intensity);
        }
        else {
            for (int step = 0; step <= steps; ++step) {
                float interpolationFactor = (steps == 0) ? 0.0f : (static_cast<float>(step) / steps);
                float interpolatedPosition = previousPosition + (currentPosition - previousPosition) * interpolationFactor;
                size_t interpolatedRowIdx = static_cast<size_t>(previousRowIdx + (currentRowIdx - previousRowIdx) * interpolationFactor);

                size_t colIdx = static_cast<size_t>(interpolatedPosition);

                // Excite the phosphor at interpolated position with proportional intensity
                if (interpolatedRowIdx < 0) {
                    interpolatedRowIdx = 0;
                }
                else if (interpolatedRowIdx >= grid.size()) {
                    interpolatedRowIdx = grid.size() - 1;
                }
                if (colIdx < grid[interpolatedRowIdx].size()) {
                    float speedFactor = 1.0f - std::min<float>(1.0f, distance / 100); // Adjust speedThreshold as needed
                    float intensityProportion = max_intensity * speedFactor;
                    grid[interpolatedRowIdx][colIdx]->excite(intensityProportion);
                }
                
            }
        }

        currentPosition += 0.5f; // Adjust this value to control the speed and smoothness
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Control frequency
    }
}

class FocusingGrid {
private:
    std::vector<std::string> frontBuffer; // The buffer currently displayed on screen
    std::vector<std::string> backBuffer;  // The buffer being prepared by the simulation thread
    std::mutex bufferMutex;
    std::thread renderThread;
    std::atomic<bool> running{ false };
    std::condition_variable cv;
    bool readyToSwap = false;
    Screen& screen; // Reference to a Screen instance

    /**
     * @brief Renders frames to the console using differential updates.
     * This method compares the new frame (in frontBuffer after the swap) with the
     * old frame (in backBuffer after the swap) and only prints the lines that
     * have changed, using ANSI escape codes to position the cursor.
     */
    void render() {
        console::clear_screen();
        std::cout << "\x1B[?25l"; // Hide cursor

        while (running) {
            std::unique_lock<std::mutex> lock(bufferMutex);
            cv.wait(lock, [this] { return readyToSwap; });

            // Swap buffers. The new frame is now in frontBuffer. The old is in backBuffer.
            std::swap(frontBuffer, backBuffer);
            readyToSwap = false;
            lock.unlock(); // Release lock before performing I/O

            for (size_t r = 0; r < frontBuffer.size(); ++r) {
                if (r >= backBuffer.size() || frontBuffer[r] != backBuffer[r]) {
                    console::move_cursor(r, 0);
                    std::cout << frontBuffer[r];
                }
            }
            std::cout << std::flush;
        }
        std::cout << "\x1B[?25h"; // Show cursor
    }
public:
    FocusingGrid(int width, int height, Screen& screenRef) : frontBuffer(height, std::string(width, ' ')), backBuffer(height, std::string(width, ' ')), screen(screenRef) {}

    void start() {
        running = true;
        renderThread = std::thread(&FocusingGrid::render, this);
    }

    void stop() {
        running = false;
        readyToSwap = true;
        cv.notify_one();
        if (renderThread.joinable()) {
            renderThread.join();
        }
    }

    void updateFromScreen(int statusBlockX, int statusBlockY) {
        std::string statusMessages = globalStatusBlock.getStatusString("\n", true);
        auto stringBuffer = screen.renderToStringBuffer(); 
        std::istringstream statusStream(statusMessages);
        std::string line;
        std::vector<std::string> statusLines;
        while (std::getline(statusStream, line)) {
            if (line.length() > console_width - statusBlockX) {
                line = line.substr(0, console_width - statusBlockX);
            }
            statusLines.push_back(line);
        }

        
        for (size_t i = 0; i < statusLines.size() && (i + statusBlockY) < stringBuffer.size(); ++i) {
            std::string& targetLine = stringBuffer[statusBlockY + i];
            targetLine.replace(statusBlockX, statusLines[i].length(), statusLines[i]);
        }

        {
            std::lock_guard<std::mutex> lock(bufferMutex);
            backBuffer = stringBuffer; // Prepare the next frame in the back buffer
            readyToSwap = true;
        }
        cv.notify_one();
    }


    ~FocusingGrid() {
        stop();
    }
};

int main(int argc, char* argv[]) {
    console::setup_utf8_console();
    console::enable_virtual_terminal_processing();
    std::atomic<bool> running{ true };

    // Default to binary mode unless --text flag is specified
    bool isBinaryMode = true;

    // Parse command-line arguments for the --text flag
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--text") {
            isBinaryMode = false;
            break;
        }
    }

    initializeGrid(); // Initializes the grid with Phosphor objects.

    Screen screen(grid); // Create a Screen instance with a reference to the grid.
    FocusingGrid focusingGrid(console_width, console_height, screen); // Pass the Screen instance to FocusingGrid

    focusingGrid.start(); // Start rendering

    std::thread inputThread([&running, isBinaryMode]() {
        inputCache.setBinaryMode(isBinaryMode);
        inputCache.processInput(running);
        });
    std::thread scannerThread(beamScan, std::ref(running));

    while (running.load()) {
        focusingGrid.updateFromScreen(1, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Clean shutdown
    running.store(false);
    inputThread.join();
    scannerThread.join();
    focusingGrid.stop();
    std::cout << "\x1B[?25h"; // Ensure cursor is visible on exit

    return 0;
}
