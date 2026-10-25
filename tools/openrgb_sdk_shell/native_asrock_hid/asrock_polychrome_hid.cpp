#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "asrock_polychrome_hid_abi.h"

#define ASROCK_API ASROCK_HID_API

static_assert(sizeof(ASRockHIDRawColor) == 3);
static_assert(sizeof(ASRockHIDExternalZoneEffect) == 16);
static_assert(sizeof(ASRockHIDUniformColor) == 16);
static_assert(offsetof(ASRockHIDExternalZoneEffect, abi_version) == 0);
static_assert(offsetof(ASRockHIDExternalZoneEffect, struct_size) == 4);
static_assert(offsetof(ASRockHIDExternalZoneEffect, mode) == 8);
static_assert(offsetof(ASRockHIDExternalZoneEffect, speed) == 9);
static_assert(offsetof(ASRockHIDExternalZoneEffect, color) == 10);
static_assert(offsetof(ASRockHIDExternalZoneEffect, reserved) == 13);
static_assert(offsetof(ASRockHIDUniformColor, abi_version) == 0);
static_assert(offsetof(ASRockHIDUniformColor, struct_size) == 4);
static_assert(offsetof(ASRockHIDUniformColor, zone_type) == 8);
static_assert(offsetof(ASRockHIDUniformColor, mode) == 9);
static_assert(offsetof(ASRockHIDUniformColor, speed) == 10);
static_assert(offsetof(ASRockHIDUniformColor, all_zones) == 11);
static_assert(offsetof(ASRockHIDUniformColor, color) == 12);
static_assert(offsetof(ASRockHIDUniformColor, reserved) == 15);

namespace {

struct hid_device_;
using hid_device = hid_device_;

struct hid_device_info {
    char* path;
    unsigned short vendor_id;
    unsigned short product_id;
    wchar_t* serial_number;
    unsigned short release_number;
    wchar_t* manufacturer_string;
    wchar_t* product_string;
    unsigned short usage_page;
    unsigned short usage;
    int interface_number;
    hid_device_info* next;
};

using HidInit = int (*)();
using HidExit = int (*)();
using HidEnumerate = hid_device_info* (*)(unsigned short, unsigned short);
using HidFreeEnumeration = void (*)(hid_device_info*);
using HidOpenPath = hid_device* (*)(const char*);
using HidWrite = int (*)(hid_device*, const unsigned char*, size_t);
using HidReadTimeout = int (*)(hid_device*, unsigned char*, size_t, int);
using HidClose = void (*)(hid_device*);
using HidError = const wchar_t* (*)(hid_device*);

constexpr unsigned char kSetZone = 0x10;
constexpr unsigned char kReadZoneConfig = 0x11;
constexpr unsigned char kWriteHeader = 0x15;
constexpr unsigned char kInitialChunk = 0xE3;
constexpr unsigned char kSendChunk = 0xE4;
constexpr unsigned char kDirectMode = 0x0F;
constexpr unsigned char kRgSwapConfig = 0x03;
constexpr size_t kReportBytes = 65;
constexpr size_t kResponseBytes = 64;
constexpr size_t kFirstPayload = 54;
constexpr size_t kLaterPayload = 57;

void set_error(char* output, size_t capacity, const std::string& message) {
    if(output == nullptr || capacity == 0) return;
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(output, message.data(), count);
    output[count] = '\0';
}

struct Transport {
    struct ScheduledFrame {
        std::vector<unsigned char> rgb;
        std::uint64_t deadline_ns = 0;
        int event_kind = 0;
    };
    struct PendingUniform {
        ASRockHIDUniformColor message{};
    };
    HMODULE library = nullptr;
    hid_device* device = nullptr;
    HidInit init = nullptr;
    HidExit exit = nullptr;
    HidEnumerate enumerate = nullptr;
    HidFreeEnumeration free_enumeration = nullptr;
    HidOpenPath open_path = nullptr;
    HidWrite write = nullptr;
    HidReadTimeout read_timeout = nullptr;
    HidClose close = nullptr;
    HidError error = nullptr;
    std::vector<unsigned char> zone_types;
    std::vector<unsigned short> zone_counts;
    std::vector<unsigned short> zone_max_counts;
    std::vector<unsigned char> zone_rg_swaps;
    std::vector<std::array<unsigned char, kReportBytes>> reports;
    std::vector<unsigned char> payload;
    size_t active_report_count = 0;
    size_t acknowledgement_window = 0;
    int message_pattern = 0;
    std::vector<std::array<unsigned char, 3>> last_colors;
    std::vector<ASRockHIDUniformColor> last_uniforms;
    std::vector<bool> have_last_uniform;
    int response_timeout_ms = 100;
    bool have_last_colors = false;
    std::mutex queue_mutex;
    std::condition_variable queue_changed;
    std::vector<unsigned char> pending_rgb;
    std::deque<ScheduledFrame> scheduled_frames;
    std::deque<PendingUniform> pending_uniforms;
    bool pending = false;
    bool busy = false;
    bool last_job_was_uniform = false;
    bool have_active_uniform = false;
    ASRockHIDUniformColor active_uniform{};
    bool have_active_frame = false;
    std::vector<unsigned char> active_rgb;
    bool stopping = false;
    std::thread worker;
    std::uint64_t submitted_frames = 0;
    std::uint64_t completed_frames = 0;
    std::uint64_t dropped_frames = 0;
    std::uint64_t failed_frames = 0;
    std::uint64_t scheduled_frames_dispatched = 0;
    std::uint64_t scheduled_on_dispatched = 0;
    std::uint64_t scheduled_off_dispatched = 0;
    int last_scheduled_event_kind = 0;
    std::int64_t last_schedule_error_ns = 0;
    std::int64_t maximum_schedule_error_ns = 0;
    std::uint64_t last_transmit_duration_ns = 0;
    std::chrono::steady_clock::time_point first_completion;
    std::chrono::steady_clock::time_point last_completion;
    std::string worker_error;

    ~Transport() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stopping = true;
        }
        queue_changed.notify_one();
        if(worker.joinable()) worker.join();
        if(device != nullptr && close != nullptr) close(device);
        if(exit != nullptr) exit();
        if(library != nullptr) FreeLibrary(library);
    }

    std::string hid_error_text(const char* prefix) const {
        std::string message(prefix);
        if(error != nullptr && device != nullptr) {
            const wchar_t* detail = error(device);
            if(detail != nullptr && *detail != L'\0') {
                char converted[512]{};
                WideCharToMultiByte(CP_UTF8, 0, detail, -1, converted,
                                    static_cast<int>(sizeof(converted)), nullptr, nullptr);
                message += ": ";
                message += converted;
            }
        }
        return message;
    }

    bool write_report(unsigned char* report, char* error_out, size_t error_capacity) {
        const int written = write(device, report, kReportBytes);
        if(written != static_cast<int>(kReportBytes)) {
            set_error(error_out, error_capacity, hid_error_text("ASRock HID write failed"));
            return false;
        }
        return true;
    }

    bool read_response(char* error_out, size_t error_capacity) {
        unsigned char response[kResponseBytes]{};
        const int received = read_timeout(device, response, sizeof(response), response_timeout_ms);
        if(received <= 0) {
            set_error(error_out, error_capacity,
                      received == 0 ? "ASRock HID response timed out"
                                    : hid_error_text("ASRock HID read failed"));
            return false;
        }
        return true;
    }

    bool read_response_into(unsigned char* response, size_t response_size,
                            char* error_out, size_t error_capacity) {
        const int received = read_timeout(device, response, response_size, response_timeout_ms);
        if(received <= 0) {
            set_error(error_out, error_capacity,
                      received == 0 ? "ASRock HID response timed out"
                                    : hid_error_text("ASRock HID read failed"));
            return false;
        }
        return true;
    }

    bool command(std::array<unsigned char, kReportBytes>& report,
                 char* error_out, size_t error_capacity) {
        return write_report(report.data(), error_out, error_capacity)
            && read_response(error_out, error_capacity);
    }

    bool transmit_frame(const std::vector<unsigned char>& rgb, size_t report_count,
                        size_t acknowledgement_batch, int pattern, std::string& failure) {
        std::vector<unsigned char> frame_payload(payload.size(), 0);
        std::vector<std::array<unsigned char, kReportBytes>> frame_reports(reports.size());
        size_t led_index = 0;
        size_t payload_index = 0;
        for(size_t zone = 0; zone < zone_counts.size(); ++zone) {
            for(size_t position = 0; position < zone_counts[zone]; ++position, ++led_index) {
                const unsigned char r = rgb[3 * led_index];
                const unsigned char g = rgb[3 * led_index + 1];
                const unsigned char b = rgb[3 * led_index + 2];
                if(zone_rg_swaps[zone]) {
                    frame_payload[payload_index] = r;
                    frame_payload[payload_index + 1] = g;
                } else {
                    frame_payload[payload_index] = g;
                    frame_payload[payload_index + 1] = r;
                }
                frame_payload[payload_index + 2] = b;
                payload_index += 3;
            }
        }

        size_t payload_cursor = 0;
        for(size_t report_index = 0; report_index < report_count; ++report_index) {
            auto& report = frame_reports[report_index];
            report.fill(0);
            report[1] = kSetZone;
            report[3] = 0xFF;
            const bool initial = pattern == 1 || (pattern == 0 && report_index == 0);
            report[4] = initial ? kInitialChunk : kSendChunk;
            report[64] = 0x65;
            const size_t offset = initial ? 9 : 5;
            const size_t capacity = initial ? kFirstPayload : kLaterPayload;
            if(initial) report[7] = 0xFF;
            const size_t available = frame_payload.size()
                - std::min(payload_cursor, frame_payload.size());
            const size_t copied = std::min(capacity, available);
            if(copied > 0) {
                std::memcpy(
                    report.data() + offset, frame_payload.data() + payload_cursor, copied);
                payload_cursor += copied;
            }
        }

        char error_buffer[1024]{};
        report_count = std::max<size_t>(1, std::min(report_count, reports.size()));
        acknowledgement_batch = std::max<size_t>(1, std::min(acknowledgement_batch, report_count));
        for(size_t first = 0; first < report_count; first += acknowledgement_batch) {
            const size_t stop = std::min(report_count, first + acknowledgement_batch);
            for(size_t index = first; index < stop; ++index) {
                if(!write_report(frame_reports[index].data(), error_buffer, sizeof(error_buffer))) {
                    failure = error_buffer;
                    return false;
                }
            }
            for(size_t index = first; index < stop; ++index) {
                if(!read_response(error_buffer, sizeof(error_buffer))) {
                    failure = error_buffer;
                    return false;
                }
            }
        }
        return true;
    }

    bool transmit_uniform(const ASRockHIDUniformColor& message, std::string& failure) {
        const auto zone = std::find(zone_types.begin(), zone_types.end(), message.zone_type);
        if(!message.all_zones && zone == zone_types.end()) {
            failure = "ASRock HID uniform color names an unavailable zone";
            return false;
        }
        unsigned char r = message.color.byte_0;
        unsigned char g = message.color.byte_1;
        const unsigned char b = message.color.byte_2;
        if(!message.all_zones) {
            const size_t zone_index = static_cast<size_t>(zone - zone_types.begin());
            if(!zone_rg_swaps[zone_index]) std::swap(r, g);
        }

        std::array<unsigned char, kReportBytes> report{};
        report[1] = kSetZone;
        report[3] = message.zone_type;
        report[4] = message.mode;
        report[5] = r;
        report[6] = g;
        report[7] = b;
        report[8] = message.speed;
        report[9] = 0xFF;
        report[16] = message.all_zones;
        char error_buffer[1024]{};
        if(!command(report, error_buffer, sizeof(error_buffer))) {
            failure = error_buffer;
            return false;
        }
        return true;
    }

    void worker_loop() {
        std::vector<unsigned char> frame;
        for(;;) {
            size_t report_count = 1;
            size_t acknowledgement_batch = 1;
            int pattern = 0;
            bool scheduled = false;
            bool uniform = false;
            ASRockHIDUniformColor uniform_message{};
            std::uint64_t scheduled_deadline_ns = 0;
            int scheduled_event_kind = 0;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_changed.wait(lock, [this] {
                    return stopping || pending || !pending_uniforms.empty()
                        || !scheduled_frames.empty();
                });
                if(stopping) {
                    scheduled_frames.clear();
                }
                if(stopping && !pending && pending_uniforms.empty()) break;
                if(!scheduled_frames.empty()) {
                    scheduled_deadline_ns = scheduled_frames.front().deadline_ns;
                    const auto deadline = std::chrono::steady_clock::time_point(
                        std::chrono::nanoseconds(scheduled_deadline_ns));
                    const auto coarse_deadline = deadline - std::chrono::microseconds(250);
                    if(std::chrono::steady_clock::now() < coarse_deadline) {
                        queue_changed.wait_until(lock, coarse_deadline, [this, scheduled_deadline_ns] {
                            return stopping || scheduled_frames.empty()
                                || scheduled_frames.front().deadline_ns != scheduled_deadline_ns;
                        });
                    }
                    if(stopping) {
                        scheduled_frames.clear();
                        if(!pending && pending_uniforms.empty()) break;
                    } else if(
                        scheduled_frames.empty()
                        || scheduled_frames.front().deadline_ns != scheduled_deadline_ns
                    ) {
                        continue;
                    } else {
                        frame = std::move(scheduled_frames.front().rgb);
                        scheduled_event_kind = scheduled_frames.front().event_kind;
                        scheduled_frames.pop_front();
                        scheduled = true;
                        active_rgb = frame;
                        have_active_frame = true;
                        busy = true;
                        report_count = active_report_count;
                        acknowledgement_batch = acknowledgement_window;
                        pattern = message_pattern;
                    }
                }
                if(
                    !scheduled && !pending_uniforms.empty()
                    && (!pending || !last_job_was_uniform)
                ) {
                    uniform_message = pending_uniforms.front().message;
                    pending_uniforms.pop_front();
                    uniform = true;
                    active_uniform = uniform_message;
                    have_active_uniform = true;
                    busy = true;
                } else if(!scheduled) {
                    frame = std::move(pending_rgb);
                    pending_rgb.clear();
                    pending = false;
                    active_rgb = frame;
                    have_active_frame = true;
                    busy = true;
                    report_count = active_report_count;
                    acknowledgement_batch = acknowledgement_window;
                    pattern = message_pattern;
                }
            }

            if(scheduled) {
                const auto deadline = std::chrono::steady_clock::time_point(
                    std::chrono::nanoseconds(scheduled_deadline_ns));
                while(std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::yield();
                }
            }
            std::string failure;
            const auto started_at = std::chrono::steady_clock::now();
            const auto schedule_error_ns = scheduled
                ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                    started_at.time_since_epoch()).count()
                    - static_cast<std::int64_t>(scheduled_deadline_ns)
                : 0;
            const bool success = uniform
                ? transmit_uniform(uniform_message, failure)
                : transmit_frame(frame, report_count, acknowledgement_batch, pattern, failure);
            const auto completed_at = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                busy = false;
                have_active_uniform = false;
                have_active_frame = false;
                last_job_was_uniform = uniform;
                last_transmit_duration_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        completed_at - started_at).count());
                if(scheduled) {
                    last_schedule_error_ns = schedule_error_ns;
                    maximum_schedule_error_ns = std::max(
                        maximum_schedule_error_ns, schedule_error_ns);
                    ++scheduled_frames_dispatched;
                    last_scheduled_event_kind = scheduled_event_kind;
                    if(scheduled_event_kind == 1) ++scheduled_on_dispatched;
                    if(scheduled_event_kind == 2) ++scheduled_off_dispatched;
                }
                if(success) {
                    if(!uniform) {
                        for(size_t index = 0; index < last_colors.size(); ++index) {
                            last_colors[index] = {
                                frame[3 * index], frame[3 * index + 1], frame[3 * index + 2]
                            };
                        }
                        have_last_colors = true;
                    } else {
                        have_last_colors = false;
                        if(uniform_message.all_zones) {
                            std::fill(have_last_uniform.begin(), have_last_uniform.end(), true);
                            std::fill(last_uniforms.begin(), last_uniforms.end(), uniform_message);
                        } else {
                            const auto zone = std::find(
                                zone_types.begin(), zone_types.end(), uniform_message.zone_type);
                            if(zone != zone_types.end()) {
                                const size_t index = static_cast<size_t>(zone - zone_types.begin());
                                last_uniforms[index] = uniform_message;
                                have_last_uniform[index] = true;
                            }
                        }
                    }
                    ++completed_frames;
                    if(completed_frames == 1) first_completion = completed_at;
                    last_completion = std::max(last_completion, completed_at);
                } else {
                    ++failed_frames;
                    worker_error = failure;
                    pending = false;
                    pending_uniforms.clear();
                    scheduled_frames.clear();
                    stopping = true;
                }
            }
            queue_changed.notify_all();
            if(!success) break;
        }
    }
};

template<typename T>
bool load_symbol(HMODULE library, const char* name, T& output,
                 char* error_out, size_t error_capacity) {
    output = reinterpret_cast<T>(GetProcAddress(library, name));
    if(output != nullptr) return true;
    set_error(error_out, error_capacity, std::string("hidapi is missing ") + name);
    return false;
}

}  // namespace

ASROCK_API void* asrock_hid_open(
    const wchar_t* hidapi_path,
    unsigned short vendor_id,
    unsigned short product_id,
    const unsigned char* zone_types,
    const unsigned short* zone_counts,
    const unsigned short* zone_max_counts,
    const unsigned char* zone_rg_swaps,
    size_t zone_count,
    int response_timeout_ms,
    char* error_out,
    size_t error_capacity) {
    if(hidapi_path == nullptr || zone_count == 0 || zone_types == nullptr ||
       zone_counts == nullptr || zone_max_counts == nullptr || zone_rg_swaps == nullptr) {
        set_error(error_out, error_capacity, "invalid ASRock HID transport configuration");
        return nullptr;
    }

    auto transport = std::make_unique<Transport>();
    transport->library = LoadLibraryW(hidapi_path);
    if(transport->library == nullptr) {
        set_error(error_out, error_capacity, "could not load hidapi-hotplug.dll");
        return nullptr;
    }
    if(!load_symbol(transport->library, "hid_init", transport->init, error_out, error_capacity) ||
       !load_symbol(transport->library, "hid_exit", transport->exit, error_out, error_capacity) ||
       !load_symbol(transport->library, "hid_enumerate", transport->enumerate, error_out, error_capacity) ||
       !load_symbol(transport->library, "hid_free_enumeration", transport->free_enumeration, error_out, error_capacity) ||
       !load_symbol(transport->library, "hid_open_path", transport->open_path, error_out, error_capacity) ||
       !load_symbol(transport->library, "hid_write", transport->write, error_out, error_capacity) ||
       !load_symbol(transport->library, "hid_read_timeout", transport->read_timeout, error_out, error_capacity) ||
       !load_symbol(transport->library, "hid_close", transport->close, error_out, error_capacity) ||
       !load_symbol(transport->library, "hid_error", transport->error, error_out, error_capacity)) {
        return nullptr;
    }
    if(transport->init() != 0) {
        set_error(error_out, error_capacity, "hidapi initialization failed");
        return nullptr;
    }

    hid_device_info* devices = transport->enumerate(vendor_id, product_id);
    for(hid_device_info* item = devices; item != nullptr; item = item->next) {
        transport->device = transport->open_path(item->path);
        if(transport->device != nullptr) break;
    }
    transport->free_enumeration(devices);
    if(transport->device == nullptr) {
        set_error(error_out, error_capacity,
                  "could not open ASRock HID 26CE:01A2; close OpenRGB and Polychrome first");
        return nullptr;
    }

    transport->zone_types.assign(zone_types, zone_types + zone_count);
    transport->zone_counts.assign(zone_counts, zone_counts + zone_count);
    transport->zone_max_counts.assign(zone_max_counts, zone_max_counts + zone_count);
    transport->zone_rg_swaps.assign(zone_rg_swaps, zone_rg_swaps + zone_count);
    transport->response_timeout_ms = std::max(1, response_timeout_ms);

    size_t logical_led_count = 0;
    size_t maximum_led_count = 0;
    for(size_t index = 0; index < zone_count; ++index) {
        if(transport->zone_counts[index] > transport->zone_max_counts[index]) {
            set_error(error_out, error_capacity, "zone count exceeds its transport maximum");
            return nullptr;
        }
        logical_led_count += transport->zone_counts[index];
        maximum_led_count += transport->zone_max_counts[index];
    }

    const size_t payload_bytes = (maximum_led_count + 8) * 3;
    const size_t stream_report_count = payload_bytes <= kFirstPayload
        ? 1
        : 1 + (payload_bytes - kFirstPayload + kLaterPayload - 1) / kLaterPayload;
    const size_t restart_report_count =
        (payload_bytes + kFirstPayload - 1) / kFirstPayload;
    transport->reports.resize(std::max(stream_report_count, restart_report_count));
    transport->payload.resize(payload_bytes, 0);
    transport->last_colors.resize(logical_led_count);
    transport->last_uniforms.resize(zone_count);
    transport->have_last_uniform.assign(zone_count, false);
    transport->active_report_count = stream_report_count;
    transport->acknowledgement_window = stream_report_count;
    transport->pending_rgb.reserve(logical_led_count * 3);
    transport->worker = std::thread([instance = transport.get()] { instance->worker_loop(); });
    return transport.release();
}

ASROCK_API int asrock_hid_select_direct(void* opaque, char* error_out, size_t error_capacity) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr) {
        set_error(error_out, error_capacity, "ASRock HID transport is not open");
        return 0;
    }

    std::array<unsigned char, kReportBytes> rg_swap{};
    rg_swap[1] = kWriteHeader;
    rg_swap[3] = kRgSwapConfig;
    rg_swap[4] = 0;
    if(!transport->command(rg_swap, error_out, error_capacity)) return 0;

    for(unsigned char zone_type : transport->zone_types) {
        std::array<unsigned char, kReportBytes> direct{};
        direct[1] = kSetZone;
        direct[3] = zone_type;
        direct[4] = kDirectMode;
        direct[8] = 0xE0;
        direct[9] = 0xFF;
        if(!transport->command(direct, error_out, error_capacity)) return 0;
    }
    return 1;
}

ASROCK_API int asrock_hid_read_external_zone_effect_message(
    void* opaque,
    ASRockHIDExternalZoneEffect* message,
    char* error_out,
    size_t error_capacity) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr || message == nullptr) {
        set_error(error_out, error_capacity, "ASRock HID transport or effect message is null");
        return 0;
    }
    constexpr unsigned char external_zone = 0x07;
    if(std::find(
        transport->zone_types.begin(), transport->zone_types.end(), external_zone)
        == transport->zone_types.end()) {
        set_error(error_out, error_capacity, "external Audio/ARGB3 zone 0x07 is unavailable");
        return 0;
    }
    std::lock_guard<std::mutex> lock(transport->queue_mutex);
    if(transport->busy || transport->pending || !transport->pending_uniforms.empty()
       || !transport->scheduled_frames.empty()) {
        set_error(error_out, error_capacity, "ASRock HID transport is active");
        return 0;
    }
    std::array<unsigned char, kReportBytes> request{};
    request[1] = kReadZoneConfig;
    request[3] = external_zone;
    if(!transport->write_report(request.data(), error_out, error_capacity)) return 0;
    unsigned char response[kResponseBytes]{};
    if(!transport->read_response_into(
        response, sizeof(response), error_out, error_capacity)) return 0;
    *message = ASRockHIDExternalZoneEffect{};
    message->abi_version = ASROCK_HID_ABI_VERSION;
    message->struct_size = sizeof(ASRockHIDExternalZoneEffect);
    message->mode = response[4];
    message->speed = response[8];
    message->color.byte_0 = response[5];
    message->color.byte_1 = response[6];
    message->color.byte_2 = response[7];
    return 1;
}

ASROCK_API int asrock_hid_write_external_zone_effect_message(
    void* opaque,
    const ASRockHIDExternalZoneEffect* message,
    char* error_out,
    size_t error_capacity) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr || message == nullptr) {
        set_error(error_out, error_capacity, "ASRock HID transport or effect message is null");
        return 0;
    }
    if(message->abi_version != ASROCK_HID_ABI_VERSION
       || message->struct_size != sizeof(ASRockHIDExternalZoneEffect)) {
        set_error(error_out, error_capacity, "ASRock HID effect message ABI does not match");
        return 0;
    }
    if(std::any_of(
        std::begin(message->reserved), std::end(message->reserved),
        [](unsigned char value) { return value != 0; })) {
        set_error(error_out, error_capacity, "ASRock HID effect message reserved bytes are nonzero");
        return 0;
    }
    constexpr unsigned char external_zone = 0x07;
    if(std::find(
        transport->zone_types.begin(), transport->zone_types.end(), external_zone)
        == transport->zone_types.end()) {
        set_error(error_out, error_capacity, "external Audio/ARGB3 zone 0x07 is unavailable");
        return 0;
    }
    std::lock_guard<std::mutex> lock(transport->queue_mutex);
    if(transport->busy || transport->pending || !transport->pending_uniforms.empty()
       || !transport->scheduled_frames.empty()) {
        set_error(error_out, error_capacity, "ASRock HID transport is active");
        return 0;
    }
    std::array<unsigned char, kReportBytes> report{};
    report[1] = kSetZone;
    report[3] = external_zone;
    report[4] = message->mode;
    report[5] = message->color.byte_0;
    report[6] = message->color.byte_1;
    report[7] = message->color.byte_2;
    report[8] = message->speed;
    report[9] = 0xFF;
    report[16] = 0x00;
    return transport->command(report, error_out, error_capacity) ? 1 : 0;
}

ASROCK_API int asrock_hid_read_external_zone_effect(
    void* opaque,
    unsigned char* mode,
    unsigned char* speed,
    unsigned char* color_byte_0,
    unsigned char* color_byte_1,
    unsigned char* color_byte_2,
    char* error_out,
    size_t error_capacity) {
    ASRockHIDExternalZoneEffect message{};
    if(!asrock_hid_read_external_zone_effect_message(
        opaque, &message, error_out, error_capacity)) return 0;
    if(mode != nullptr) *mode = message.mode;
    if(speed != nullptr) *speed = message.speed;
    if(color_byte_0 != nullptr) *color_byte_0 = message.color.byte_0;
    if(color_byte_1 != nullptr) *color_byte_1 = message.color.byte_1;
    if(color_byte_2 != nullptr) *color_byte_2 = message.color.byte_2;
    return 1;
}

ASROCK_API int asrock_hid_write_external_zone_effect(
    void* opaque,
    unsigned char mode,
    unsigned char speed,
    unsigned char color_byte_0,
    unsigned char color_byte_1,
    unsigned char color_byte_2,
    char* error_out,
    size_t error_capacity) {
    ASRockHIDExternalZoneEffect message{};
    message.abi_version = ASROCK_HID_ABI_VERSION;
    message.struct_size = sizeof(message);
    message.mode = mode;
    message.speed = speed;
    message.color = {color_byte_0, color_byte_1, color_byte_2};
    return asrock_hid_write_external_zone_effect_message(
        opaque, &message, error_out, error_capacity);
}

ASROCK_API int asrock_hid_send_frame(
    void* opaque,
    const unsigned char* rgb,
    size_t led_count,
    char* error_out,
    size_t error_capacity) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr || rgb == nullptr || led_count != transport->last_colors.size()) {
        set_error(error_out, error_capacity, "ASRock HID frame has the wrong LED count");
        return 0;
    }

    {
        std::lock_guard<std::mutex> lock(transport->queue_mutex);
        if(!transport->worker_error.empty()) {
            set_error(error_out, error_capacity, transport->worker_error);
            return 0;
        }
        if(transport->stopping) {
            set_error(error_out, error_capacity, "ASRock HID transport is closing");
            return 0;
        }
        ++transport->submitted_frames;
        if(
            !transport->pending && transport->have_active_frame
            && transport->active_rgb.size() == led_count * 3
            && std::equal(transport->active_rgb.begin(), transport->active_rgb.end(), rgb)
        ) {
            ++transport->dropped_frames;
            return 1;
        }
        if(
            transport->pending
            && transport->pending_rgb.size() == led_count * 3
            && std::equal(transport->pending_rgb.begin(), transport->pending_rgb.end(), rgb)
        ) {
            ++transport->dropped_frames;
            return 1;
        }
        if(
            !transport->busy && !transport->pending
            && transport->pending_uniforms.empty()
            && transport->scheduled_frames.empty() && transport->have_last_colors
        ) {
            bool unchanged = true;
            for(size_t index = 0; index < led_count && unchanged; ++index) {
                unchanged = transport->last_colors[index][0] == rgb[3 * index]
                    && transport->last_colors[index][1] == rgb[3 * index + 1]
                    && transport->last_colors[index][2] == rgb[3 * index + 2];
            }
            if(unchanged) {
                ++transport->dropped_frames;
                return 1;
            }
        }
        if(transport->pending) ++transport->dropped_frames;
        transport->pending_rgb.assign(rgb, rgb + led_count * 3);
        transport->pending = true;
    }
    transport->queue_changed.notify_one();
    return 1;
}

ASROCK_API int asrock_hid_send_uniform_color(
    void* opaque,
    const ASRockHIDUniformColor* message,
    char* error_out,
    size_t error_capacity) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr || message == nullptr) {
        set_error(error_out, error_capacity, "ASRock HID transport or uniform message is null");
        return 0;
    }
    if(message->abi_version != ASROCK_HID_ABI_VERSION
       || message->struct_size != sizeof(ASRockHIDUniformColor)) {
        set_error(error_out, error_capacity, "ASRock HID uniform message ABI does not match");
        return 0;
    }
    if(message->mode > kDirectMode) {
        set_error(error_out, error_capacity, "ASRock HID uniform mode is outside the known range");
        return 0;
    }
    if(message->all_zones > 1) {
        set_error(error_out, error_capacity, "ASRock HID uniform all_zones must be zero or one");
        return 0;
    }
    if(std::any_of(
        std::begin(message->reserved), std::end(message->reserved),
        [](unsigned char value) { return value != 0; })) {
        set_error(error_out, error_capacity, "ASRock HID uniform message reserved bytes are nonzero");
        return 0;
    }
    if(!message->all_zones && std::find(
        transport->zone_types.begin(), transport->zone_types.end(), message->zone_type)
        == transport->zone_types.end()) {
        set_error(error_out, error_capacity, "ASRock HID uniform message names an unavailable zone");
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(transport->queue_mutex);
        if(!transport->worker_error.empty()) {
            set_error(error_out, error_capacity, transport->worker_error);
            return 0;
        }
        if(transport->stopping) {
            set_error(error_out, error_capacity, "ASRock HID transport is closing");
            return 0;
        }
        ++transport->submitted_frames;
        if(transport->have_active_uniform
           && transport->active_uniform.zone_type == message->zone_type
           && std::none_of(
               transport->pending_uniforms.begin(), transport->pending_uniforms.end(),
               [message](const Transport::PendingUniform& item) {
                   return item.message.zone_type == message->zone_type;
               })
           && std::memcmp(&transport->active_uniform, message, sizeof(*message)) == 0) {
            ++transport->dropped_frames;
            return 1;
        }
        if(!message->all_zones) {
            const auto zone = std::find(
                transport->zone_types.begin(), transport->zone_types.end(), message->zone_type);
            const size_t zone_index = static_cast<size_t>(zone - transport->zone_types.begin());
            const bool same_zone_active = transport->have_active_uniform
                && transport->active_uniform.zone_type == message->zone_type;
            const bool same_zone_pending = std::any_of(
                transport->pending_uniforms.begin(), transport->pending_uniforms.end(),
                [message](const Transport::PendingUniform& item) {
                    return item.message.zone_type == message->zone_type;
                });
            // Activity in another physical zone must not invalidate this zone's
            // resident native mode.  A controller-wide direct frame still does.
            if(!same_zone_active && !same_zone_pending
               && !transport->pending && !transport->have_active_frame
               && transport->scheduled_frames.empty()
               && transport->have_last_uniform[zone_index]
               && std::memcmp(
                   &transport->last_uniforms[zone_index], message, sizeof(*message)) == 0) {
                ++transport->dropped_frames;
                return 1;
            }
        }
        const auto same_zone = std::find_if(
            transport->pending_uniforms.begin(), transport->pending_uniforms.end(),
            [message](const Transport::PendingUniform& item) {
                return item.message.zone_type == message->zone_type;
            });
        if(same_zone != transport->pending_uniforms.end()) {
            if(std::memcmp(&same_zone->message, message, sizeof(*message)) == 0) {
                ++transport->dropped_frames;
                return 1;
            }
            same_zone->message = *message;
            ++transport->dropped_frames;
        } else {
            transport->pending_uniforms.push_back(Transport::PendingUniform{*message});
        }
    }
    transport->queue_changed.notify_one();
    return 1;
}

ASROCK_API std::uint64_t asrock_hid_clock_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

ASROCK_API int asrock_hid_send_frame_at(
    void* opaque,
    const unsigned char* rgb,
    size_t led_count,
    std::uint64_t deadline_ns,
    int event_kind,
    char* error_out,
    size_t error_capacity) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr || rgb == nullptr || led_count != transport->last_colors.size()) {
        set_error(error_out, error_capacity, "ASRock HID scheduled frame has the wrong LED count");
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(transport->queue_mutex);
        if(!transport->worker_error.empty()) {
            set_error(error_out, error_capacity, transport->worker_error);
            return 0;
        }
        if(transport->stopping) {
            set_error(error_out, error_capacity, "ASRock HID transport is closing");
            return 0;
        }
        if(transport->scheduled_frames.size() >= 16) {
            set_error(error_out, error_capacity, "ASRock HID scheduled queue is full");
            return 0;
        }
        Transport::ScheduledFrame scheduled;
        scheduled.rgb.assign(rgb, rgb + led_count * 3);
        scheduled.deadline_ns = deadline_ns;
        scheduled.event_kind = event_kind;
        const auto position = std::upper_bound(
            transport->scheduled_frames.begin(), transport->scheduled_frames.end(), deadline_ns,
            [](std::uint64_t value, const Transport::ScheduledFrame& item) {
                return value < item.deadline_ns;
            });
        transport->scheduled_frames.insert(position, std::move(scheduled));
        ++transport->submitted_frames;
    }
    transport->queue_changed.notify_one();
    return 1;
}

ASROCK_API void asrock_hid_cancel_scheduled(void* opaque) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr) return;
    {
        std::lock_guard<std::mutex> lock(transport->queue_mutex);
        transport->scheduled_frames.clear();
    }
    transport->queue_changed.notify_all();
}

ASROCK_API int asrock_hid_timing_metrics(
    void* opaque,
    std::uint64_t* dispatched,
    std::int64_t* last_error_ns,
    std::int64_t* maximum_error_ns,
    std::uint64_t* transmit_duration_ns,
    size_t* queued,
    std::uint64_t* on_dispatched,
    std::uint64_t* off_dispatched,
    int* last_event_kind) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr) return 0;
    std::lock_guard<std::mutex> lock(transport->queue_mutex);
    if(dispatched != nullptr) *dispatched = transport->scheduled_frames_dispatched;
    if(last_error_ns != nullptr) *last_error_ns = transport->last_schedule_error_ns;
    if(maximum_error_ns != nullptr) *maximum_error_ns = transport->maximum_schedule_error_ns;
    if(transmit_duration_ns != nullptr) *transmit_duration_ns = transport->last_transmit_duration_ns;
    if(queued != nullptr) *queued = transport->scheduled_frames.size();
    if(on_dispatched != nullptr) *on_dispatched = transport->scheduled_on_dispatched;
    if(off_dispatched != nullptr) *off_dispatched = transport->scheduled_off_dispatched;
    if(last_event_kind != nullptr) *last_event_kind = transport->last_scheduled_event_kind;
    return 1;
}

ASROCK_API int asrock_hid_metrics(
    void* opaque,
    std::uint64_t* submitted,
    std::uint64_t* completed,
    std::uint64_t* dropped,
    std::uint64_t* failed,
    int* busy,
    int* pending,
    double* completion_hz,
    size_t* active_report_count,
    size_t* acknowledgement_window,
    int* message_pattern) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr) return 0;
    std::lock_guard<std::mutex> lock(transport->queue_mutex);
    if(submitted != nullptr) *submitted = transport->submitted_frames;
    if(completed != nullptr) *completed = transport->completed_frames;
    if(dropped != nullptr) *dropped = transport->dropped_frames;
    if(failed != nullptr) *failed = transport->failed_frames;
    if(busy != nullptr) *busy = transport->busy ? 1 : 0;
    if(pending != nullptr) *pending = transport->pending ? 1 : 0;
    if(active_report_count != nullptr) *active_report_count = transport->active_report_count;
    if(acknowledgement_window != nullptr) *acknowledgement_window = transport->acknowledgement_window;
    if(message_pattern != nullptr) *message_pattern = transport->message_pattern;
    if(completion_hz != nullptr) {
        *completion_hz = 0.0;
        if(transport->completed_frames > 1) {
            const double seconds = std::chrono::duration<double>(
                transport->last_completion - transport->first_completion).count();
            if(seconds > 0.0) {
                *completion_hz = static_cast<double>(transport->completed_frames - 1) / seconds;
            }
        }
    }
    return 1;
}

ASROCK_API int asrock_hid_configure_packets(
    void* opaque,
    size_t report_count,
    size_t acknowledgement_window,
    int message_pattern,
    char* error_out,
    size_t error_capacity) {
    auto* transport = static_cast<Transport*>(opaque);
    if(transport == nullptr) {
        set_error(error_out, error_capacity, "ASRock HID transport is not open");
        return 0;
    }
    if(report_count == 0 || report_count > transport->reports.size()) {
        set_error(error_out, error_capacity, "report prefix is outside the available range");
        return 0;
    }
    if(acknowledgement_window == 0 || acknowledgement_window > report_count) {
        set_error(error_out, error_capacity, "acknowledgement window is outside the report prefix");
        return 0;
    }
    if(message_pattern < 0 || message_pattern > 2) {
        set_error(error_out, error_capacity, "message pattern is outside the available range");
        return 0;
    }
    std::lock_guard<std::mutex> lock(transport->queue_mutex);
    transport->active_report_count = report_count;
    transport->acknowledgement_window = acknowledgement_window;
    transport->message_pattern = message_pattern;
    return 1;
}

ASROCK_API size_t asrock_hid_report_count(void* opaque) {
    const auto* transport = static_cast<const Transport*>(opaque);
    return transport == nullptr ? 0 : transport->reports.size();
}

ASROCK_API void asrock_hid_close(void* opaque) {
    delete static_cast<Transport*>(opaque);
}
