#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#  define ASROCK_HID_EXTERN_C extern "C"
#else
#  define ASROCK_HID_EXTERN_C extern
#endif

#if defined(_WIN32)
#  if defined(ASROCK_HID_EXPORTS)
#    define ASROCK_HID_API ASROCK_HID_EXTERN_C __declspec(dllexport)
#  else
#    define ASROCK_HID_API ASROCK_HID_EXTERN_C __declspec(dllimport)
#  endif
#else
#  define ASROCK_HID_API ASROCK_HID_EXTERN_C
#endif

#define ASROCK_HID_ABI_VERSION 2u

/* Raw controller color order.  The semantic ABI preserves these bytes so a
 * read followed by a write restores exactly what command 0x11 returned. */
typedef struct ASRockHIDRawColor
{
    uint8_t byte_0;
    uint8_t byte_1;
    uint8_t byte_2;
} ASRockHIDRawColor;

/* Semantic message for the deliberately bounded Audio/ARGB3 (zone 0x07)
 * effect ABI.  The native library remains responsible for constructing and
 * validating the physical 65-byte HID report. */
typedef struct ASRockHIDExternalZoneEffect
{
    uint32_t            abi_version;
    uint32_t            struct_size;
    uint8_t             mode;
    uint8_t             speed;
    ASRockHIDRawColor   color;
    uint8_t             reserved[3];
} ASRockHIDExternalZoneEffect;

/* One configured physical zone, driven by one known native hardware mode and
 * one semantic RGB value. DIRECT (0x0F) hands the zone back to frame streaming. */
typedef struct ASRockHIDUniformColor
{
    uint32_t            abi_version;
    uint32_t            struct_size;
    uint8_t             zone_type;
    uint8_t             mode;
    uint8_t             speed;
    uint8_t             all_zones;
    ASRockHIDRawColor   color;
    uint8_t             reserved[1];
} ASRockHIDUniformColor;

ASROCK_HID_API int asrock_hid_read_external_zone_effect_message(
    void* transport,
    ASRockHIDExternalZoneEffect* message,
    char* error_out,
    size_t error_capacity);

ASROCK_HID_API int asrock_hid_write_external_zone_effect_message(
    void* transport,
    const ASRockHIDExternalZoneEffect* message,
    char* error_out,
    size_t error_capacity);

ASROCK_HID_API int asrock_hid_send_uniform_color(
    void* transport,
    const ASRockHIDUniformColor* message,
    char* error_out,
    size_t error_capacity);
