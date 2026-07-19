# Third-Party Dependencies

This directory vendors all external C libraries used by the low-level
implementations of the `asciioscilloscope` project under the monorepo.
Each subfolder corresponds to a project pinned to a specific commit or release.

Whenever possible, keep the sources here so that builds are entirely
offline and reproducible. If a dependency is missing, the CMake build
script will attempt to fetch the required source code on demand.

## Git Addresses of Components

- Eigen: https://gitlab.com/libeigen/eigen
- libigl: https://github.com/libigl/libigl
- imgui: https://github.com/ocornut/imgui
- glfw: https://github.com/glfw/glfw
- glad: https://github.com/Dav1dde/glad
- stb: https://github.com/nothings/stb
- tinydnn: https://github.com/tiny-dnn/tiny-dnn
- onnxruntime: https://github.com/microsoft/onnxruntime
- libdec: [Custom or self-maintained repository]
- freetype: https://gitlab.freedesktop.org/freetype/freetype
- fontconfig: [Optional, if dynamic font discovery is needed]
- nanogui: https://github.com/wjakob/nanogui
