#define NOMINMAX
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

#if defined(OPENRGB_AUDIO_SCOPE_EXPORTS)
#define SCOPE_API __declspec(dllexport)
#else
#define SCOPE_API __declspec(dllimport)
#endif

using fft_init_fn = void* (__cdecl *)(size_t, int, int, int, int, int, const int*, size_t, int);
using fft_execute_fn = int (__cdecl *)(void*, const float*, float*, float*, float*, size_t);
using fft_free_fn = void (__cdecl *)(void*);

namespace {

constexpr size_t kCaptureCapacity = 1u << 18;
constexpr float kPi = 3.14159265358979323846f;

bool power_of_two(size_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

bool valid_fft_size(size_t value) {
  return value >= 256 && power_of_two(value);
}

struct ScopeContext {
  HMODULE fft_module = nullptr;
  fft_init_fn fft_init = nullptr;
  fft_execute_fn fft_execute = nullptr;
  fft_free_fn fft_free = nullptr;
  void* fft_left = nullptr;
  void* fft_right = nullptr;

  std::atomic<bool> stop{false};
  std::atomic<bool> running{false};
  std::atomic<uint32_t> sample_rate{0};
  std::atomic<uint32_t> channel_count{0};
  std::thread capture_thread;

  std::mutex sample_mutex;
  std::vector<float> left = std::vector<float>(kCaptureCapacity, 0.0f);
  std::vector<float> right = std::vector<float>(kCaptureCapacity, 0.0f);
  size_t write_index = 0;
  size_t available = 0;
  uint64_t total_frames = 0;

  std::mutex analysis_mutex;
  size_t fft_size = 4096;
  size_t oversample = 8;
  size_t band_count = 128;
  float min_hz = 27.5f;
  float max_hz = 18000.0f;
  uint64_t last_analysis_frame = 0;
  std::vector<float> cached_xy;
  std::vector<float> cached_power;
  std::vector<float> cached_frequency;
  float cached_rms = 0.0f;

  std::mutex error_mutex;
  std::string error;

  void set_error(const std::string& text) {
    std::lock_guard<std::mutex> lock(error_mutex);
    error = text;
  }

  bool make_plans(size_t size) {
    if (!valid_fft_size(size) || !fft_init || !fft_free) return false;
    void* next_left = fft_init(size, 1, 0, 0, 0, 0, nullptr, 0, 2);
    void* next_right = fft_init(size, 1, 0, 0, 0, 0, nullptr, 0, 2);
    if (!next_left || !next_right) {
      if (next_left) fft_free(next_left);
      if (next_right) fft_free(next_right);
      return false;
    }
    if (fft_left) fft_free(fft_left);
    if (fft_right) fft_free(fft_right);
    fft_left = next_left;
    fft_right = next_right;
    return true;
  }

  ~ScopeContext() {
    stop.store(true);
    if (capture_thread.joinable()) capture_thread.join();
    if (fft_left && fft_free) fft_free(fft_left);
    if (fft_right && fft_free) fft_free(fft_right);
    if (fft_module) FreeLibrary(fft_module);
  }
};

float pcm_sample(const BYTE* source, WORD tag, WORD bits) {
  if (tag == WAVE_FORMAT_IEEE_FLOAT && bits == 32) {
    float value;
    std::memcpy(&value, source, sizeof(value));
    return value;
  }
  if (tag == WAVE_FORMAT_PCM && bits == 16) {
    int16_t value;
    std::memcpy(&value, source, sizeof(value));
    return static_cast<float>(value) / 32768.0f;
  }
  if (tag == WAVE_FORMAT_PCM && bits == 24) {
    int32_t value = static_cast<int32_t>(source[0])
                  | (static_cast<int32_t>(source[1]) << 8)
                  | (static_cast<int32_t>(source[2]) << 16);
    if (value & 0x00800000) value |= ~0x00ffffff;
    return static_cast<float>(value) / 8388608.0f;
  }
  if (tag == WAVE_FORMAT_PCM && bits == 32) {
    int32_t value;
    std::memcpy(&value, source, sizeof(value));
    return static_cast<float>(value) / 2147483648.0f;
  }
  return 0.0f;
}

void capture_worker(ScopeContext* context) {
  const HRESULT com_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  const bool com_ready = SUCCEEDED(com_result);
  if (!com_ready) {
    context->set_error("CoInitializeEx failed in native WASAPI worker");
    return;
  }

  ComPtr<IMMDeviceEnumerator> enumerator;
  ComPtr<IMMDevice> endpoint;
  ComPtr<IAudioClient> client;
  ComPtr<IAudioCaptureClient> capture;
  WAVEFORMATEX* format = nullptr;
  HRESULT hr = CoCreateInstance(
      __uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
      IID_PPV_ARGS(enumerator.GetAddressOf()));
  if (SUCCEEDED(hr)) hr = enumerator->GetDefaultAudioEndpoint(eRender, eMultimedia, endpoint.GetAddressOf());
  if (SUCCEEDED(hr)) hr = endpoint->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                                              reinterpret_cast<void**>(client.GetAddressOf()));
  if (SUCCEEDED(hr)) hr = client->GetMixFormat(&format);
  if (FAILED(hr) || !format) {
    context->set_error("could not open the default Windows speaker endpoint");
    if (format) CoTaskMemFree(format);
    CoUninitialize();
    return;
  }

  WORD sample_tag = format->wFormatTag;
  if (sample_tag == WAVE_FORMAT_EXTENSIBLE && format->cbSize >= 22) {
    auto* extended = reinterpret_cast<WAVEFORMATEXTENSIBLE*>(format);
    sample_tag = static_cast<WORD>(extended->SubFormat.Data1 & 0xffffu);
  }
  if (format->nChannels < 2 ||
      !((sample_tag == WAVE_FORMAT_IEEE_FLOAT && format->wBitsPerSample == 32) ||
        (sample_tag == WAVE_FORMAT_PCM &&
         (format->wBitsPerSample == 16 || format->wBitsPerSample == 24 || format->wBitsPerSample == 32)))) {
    context->set_error("unsupported native WASAPI speaker mix format");
    CoTaskMemFree(format);
    CoUninitialize();
    return;
  }

  hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK,
                          10000000, 0, format, nullptr);
  if (SUCCEEDED(hr)) hr = client->GetService(IID_PPV_ARGS(capture.GetAddressOf()));
  if (SUCCEEDED(hr)) hr = client->Start();
  if (FAILED(hr)) {
    context->set_error("native WASAPI loopback initialization failed");
    CoTaskMemFree(format);
    CoUninitialize();
    return;
  }

  context->sample_rate.store(format->nSamplesPerSec);
  context->channel_count.store(format->nChannels);
  context->running.store(true);
  const size_t bytes_per_sample = format->wBitsPerSample / 8;

  while (!context->stop.load()) {
    UINT32 packet_frames = 0;
    if (FAILED(capture->GetNextPacketSize(&packet_frames))) break;
    if (packet_frames == 0) {
      Sleep(2);
      continue;
    }
    while (packet_frames > 0) {
      BYTE* data = nullptr;
      UINT32 frames = 0;
      DWORD flags = 0;
      if (FAILED(capture->GetBuffer(&data, &frames, &flags, nullptr, nullptr))) {
        packet_frames = 0;
        break;
      }
      {
        std::lock_guard<std::mutex> lock(context->sample_mutex);
        for (UINT32 frame = 0; frame < frames; ++frame) {
          float left = 0.0f;
          float right = 0.0f;
          if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
            const BYTE* row = data + static_cast<size_t>(frame) * format->nBlockAlign;
            left = pcm_sample(row, sample_tag, format->wBitsPerSample);
            right = pcm_sample(row + bytes_per_sample, sample_tag, format->wBitsPerSample);
          }
          context->left[context->write_index] = left;
          context->right[context->write_index] = right;
          context->write_index = (context->write_index + 1) % kCaptureCapacity;
          context->available = std::min(kCaptureCapacity, context->available + 1);
          ++context->total_frames;
        }
      }
      capture->ReleaseBuffer(frames);
      if (FAILED(capture->GetNextPacketSize(&packet_frames))) packet_frames = 0;
    }
  }

  context->running.store(false);
  client->Stop();
  CoTaskMemFree(format);
  CoUninitialize();
}

bool analyze(ScopeContext* context) {
  const uint32_t sample_rate = context->sample_rate.load();
  if (!sample_rate || !context->fft_execute || !context->fft_left || !context->fft_right) return false;

  std::vector<float> left(context->fft_size, 0.0f);
  std::vector<float> right(context->fft_size, 0.0f);
  uint64_t total_frames = 0;
  {
    std::lock_guard<std::mutex> lock(context->sample_mutex);
    total_frames = context->total_frames;
    const size_t copy_count = std::min(context->available, context->fft_size);
    const size_t destination = context->fft_size - copy_count;
    const size_t start = (context->write_index + kCaptureCapacity - copy_count) % kCaptureCapacity;
    for (size_t index = 0; index < copy_count; ++index) {
      const size_t source = (start + index) % kCaptureCapacity;
      left[destination + index] = context->left[source];
      right[destination + index] = context->right[source];
    }
  }

  const size_t hop = std::max<size_t>(1, context->fft_size / context->oversample);
  if (!context->cached_power.empty() && total_frames - context->last_analysis_frame < hop) {
    return true;
  }
  context->last_analysis_frame = total_frames;

  double rms_sum = 0.0;
  double window_sum = 0.0;
  for (size_t index = 0; index < context->fft_size; ++index) {
    const float window = 0.5f - 0.5f * std::cos(
        2.0f * kPi * static_cast<float>(index) / static_cast<float>(context->fft_size - 1));
    rms_sum += 0.5 * (static_cast<double>(left[index]) * left[index]
                   + static_cast<double>(right[index]) * right[index]);
    left[index] *= window;
    right[index] *= window;
    window_sum += window;
  }
  context->cached_rms = static_cast<float>(std::sqrt(rms_sum / context->fft_size));

  std::vector<float> left_real(context->fft_size), left_imag(context->fft_size), left_mag(context->fft_size);
  std::vector<float> right_real(context->fft_size), right_imag(context->fft_size), right_mag(context->fft_size);
  if (!context->fft_execute(context->fft_left, left.data(), left_real.data(), left_imag.data(),
                            left_mag.data(), context->fft_size) ||
      !context->fft_execute(context->fft_right, right.data(), right_real.data(), right_imag.data(),
                            right_mag.data(), context->fft_size)) {
    context->set_error("fftfree fft_execute failed");
    return false;
  }

  const float nyquist = 0.5f * sample_rate;
  const float low = std::max(1.0f, context->min_hz);
  const float high = std::min(std::max(low * 1.001f, context->max_hz), nyquist);
  const double log_low = std::log(low);
  const double log_span = std::log(high) - log_low;
  const float coefficient_scale = static_cast<float>(2.0 / std::max(1.0, window_sum));
  context->cached_xy.assign(context->band_count * 2, 0.0f);
  context->cached_power.assign(context->band_count, 0.0f);
  context->cached_frequency.assign(context->band_count, 0.0f);

  for (size_t band = 0; band < context->band_count; ++band) {
    const double edge0 = std::exp(log_low + log_span * static_cast<double>(band) / context->band_count);
    const double edge1 = std::exp(log_low + log_span * static_cast<double>(band + 1) / context->band_count);
    size_t first = std::max<size_t>(1, static_cast<size_t>(std::ceil(edge0 * context->fft_size / sample_rate)));
    size_t last = std::min(context->fft_size / 2,
                           static_cast<size_t>(std::floor(edge1 * context->fft_size / sample_rate)));
    if (last < first) {
      first = std::max<size_t>(1, static_cast<size_t>(std::llround(
          std::sqrt(edge0 * edge1) * context->fft_size / sample_rate)));
      last = std::min(context->fft_size / 2, first);
    }

    double left_center = 0.0;
    double right_center = 0.0;
    double power_sum = 0.0;
    for (size_t bin = first; bin <= last; ++bin) {
      const double center_phase = (bin & 1u) ? -1.0 : 1.0;
      left_center += center_phase * left_real[bin];
      right_center += center_phase * right_real[bin];
      power_sum += 0.5 * (static_cast<double>(left_mag[bin]) * left_mag[bin]
                        + static_cast<double>(right_mag[bin]) * right_mag[bin]);
    }
    const double band_amplitude = coefficient_scale * std::sqrt(power_sum);
    const double db = 20.0 * std::log10(std::max(1.0e-9, band_amplitude));
    const float brightness = static_cast<float>(std::clamp((db + 80.0) / 80.0, 0.0, 1.0));
    const float left_value = coefficient_scale * static_cast<float>(left_center);
    const float right_value = coefficient_scale * static_cast<float>(right_center);
    context->cached_xy[2 * band] = 0.70710678118f * (left_value + right_value);
    context->cached_xy[2 * band + 1] = 0.70710678118f * (left_value - right_value);
    context->cached_power[band] = brightness;
    context->cached_frequency[band] = static_cast<float>(std::sqrt(edge0 * edge1));
  }
  return true;
}

}  // namespace

extern "C" {

SCOPE_API void* openrgb_audio_scope_create(
    const wchar_t* fftfree_dll, size_t fft_size, size_t oversample,
    size_t band_count, float min_hz, float max_hz) {
  if (!fftfree_dll || !valid_fft_size(fft_size) || !power_of_two(oversample) ||
      oversample > fft_size || band_count == 0 || !(min_hz > 0.0f) || !(max_hz > min_hz)) {
    return nullptr;
  }
  auto context = std::make_unique<ScopeContext>();
  context->fft_module = LoadLibraryW(fftfree_dll);
  if (!context->fft_module) return nullptr;
  context->fft_init = reinterpret_cast<fft_init_fn>(GetProcAddress(context->fft_module, "fft_init"));
  context->fft_execute = reinterpret_cast<fft_execute_fn>(GetProcAddress(context->fft_module, "fft_execute"));
  context->fft_free = reinterpret_cast<fft_free_fn>(GetProcAddress(context->fft_module, "fft_free"));
  if (!context->fft_init || !context->fft_execute || !context->fft_free || !context->make_plans(fft_size)) {
    return nullptr;
  }
  context->fft_size = fft_size;
  context->oversample = oversample;
  context->band_count = band_count;
  context->min_hz = min_hz;
  context->max_hz = max_hz;
  return context.release();
}

SCOPE_API int openrgb_audio_scope_start(void* handle) {
  auto* context = static_cast<ScopeContext*>(handle);
  if (!context || context->capture_thread.joinable()) return 0;
  context->capture_thread = std::thread(capture_worker, context);
  return 1;
}

SCOPE_API int openrgb_audio_scope_configure(
    void* handle, size_t fft_size, size_t oversample, size_t band_count,
    float min_hz, float max_hz) {
  auto* context = static_cast<ScopeContext*>(handle);
  if (!context || !valid_fft_size(fft_size) || !power_of_two(oversample) ||
      oversample > fft_size || band_count == 0 || !(min_hz > 0.0f) || !(max_hz > min_hz)) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(context->analysis_mutex);
  if (fft_size != context->fft_size && !context->make_plans(fft_size)) return 0;
  context->fft_size = fft_size;
  context->oversample = oversample;
  context->band_count = band_count;
  context->min_hz = min_hz;
  context->max_hz = max_hz;
  context->last_analysis_frame = 0;
  context->cached_power.clear();
  return 1;
}

SCOPE_API size_t openrgb_audio_scope_snapshot(
    void* handle, float gain, float* out_xy, float* out_power,
    float* out_frequency, size_t capacity, float* out_rms,
    uint32_t* out_sample_rate) {
  auto* context = static_cast<ScopeContext*>(handle);
  if (!context || !out_xy || !out_power || !out_frequency || !out_rms ||
      !out_sample_rate || capacity < context->band_count) return 0;
  std::lock_guard<std::mutex> lock(context->analysis_mutex);
  if (!analyze(context)) return 0;
  std::transform(context->cached_xy.begin(), context->cached_xy.end(), out_xy,
                 [gain](float value) { return gain * value; });
  std::copy(context->cached_power.begin(), context->cached_power.end(), out_power);
  std::copy(context->cached_frequency.begin(), context->cached_frequency.end(), out_frequency);
  *out_rms = context->cached_rms;
  *out_sample_rate = context->sample_rate.load();
  return context->band_count;
}

SCOPE_API int openrgb_audio_scope_running(void* handle) {
  auto* context = static_cast<ScopeContext*>(handle);
  return context && context->running.load() ? 1 : 0;
}

SCOPE_API const char* openrgb_audio_scope_error(void* handle) {
  auto* context = static_cast<ScopeContext*>(handle);
  if (!context) return "native audio scope handle is null";
  std::lock_guard<std::mutex> lock(context->error_mutex);
  return context->error.c_str();
}

SCOPE_API void openrgb_audio_scope_free(void* handle) {
  delete static_cast<ScopeContext*>(handle);
}

}
