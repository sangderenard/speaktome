#!/bin/bash
set -euo pipefail

ROOT_DIR="$(dirname "$(readlink -f "$0")")"
THIRD_PARTY_DIR="$ROOT_DIR/asciioscilloscope/third_party"

# Format: folder_name git_url
DEPENDENCIES=(
  "eigen https://gitlab.com/libeigen/eigen.git"
  "libigl https://github.com/libigl/libigl.git"
  "imgui https://github.com/ocornut/imgui.git"
  "glfw https://github.com/glfw/glfw.git"
  "glad https://github.com/Dav1dde/glad.git"
  "stb https://github.com/nothings/stb.git"
  "tinydnn https://github.com/tiny-dnn/tiny-dnn.git"
  "onnxruntime https://github.com/microsoft/onnxruntime.git"
  "libdec https://github.com/hkourkchi/libdec.git"
  "freetype https://gitlab.freedesktop.org/freetype/freetype.git"
  "fontconfig https://gitlab.freedesktop.org/fontconfig/fontconfig.git"
  "nanogui https://github.com/wjakob/nanogui.git"
  "assimp https://github.com/assimp/assimp.git"
  "libpng https://github.com/glennrp/libpng.git"
  "msdfgen https://github.com/Chlumsky/msdfgen.git"
)

mkdir -p "$THIRD_PARTY_DIR"
cd "$ROOT_DIR"

# Initialize .gitmodules if it doesn't exist
if [ ! -f .gitmodules ]; then
  echo "# Auto-generated .gitmodules" > .gitmodules
fi

for entry in "${DEPENDENCIES[@]}"; do
  read -r name url <<< "$entry"
  path="asciioscilloscope/third_party/$name"

  echo "? Checking submodule: $name"

  # Remove previous [submodule "x"] section from .gitmodules if duplicated
  if git config -f .gitmodules --get submodule."$path".url >/dev/null; then
    git config -f .gitmodules --remove-section submodule."$path" || true
  fi

  # Add submodule if not present
  if ! git submodule status "$path" >/dev/null 2>&1; then
    # Dynamically detect default branch
    default_branch=$(git ls-remote --symref "$url" HEAD | awk '/^ref:/ {sub("refs/heads/", "", $2); print $2}')
    echo "?? Detected default branch for $name: $default_branch"
    git submodule add -f -b "$default_branch" "$url" "$path"
  fi

  # Prevent pushing to submodule origin
  git config -f .gitmodules submodule."$path".update none
done

# Write changes to .gitmodules into .git/config
git submodule sync --recursive

# Backup, clean, and fetch submodules safely
for entry in "${DEPENDENCIES[@]}"; do
  read -r name url <<< "$entry"
  folder="$THIRD_PARTY_DIR/$name"

  if [ -d "$folder" ]; then
    echo "?? Verifying existing $name..."
    if [ ! -d "$folder/.git" ]; then
      echo "??  $name exists but is not a valid submodule. Backing up and retrying."
      mv "$folder" "${folder}-bak-$(date +%s)"
      git submodule update --init --recursive "$folder"
    else
      git submodule update --init --recursive "$folder"
    fi
  else
    echo "??  Fetching fresh: $name"
    git submodule update --init --recursive "$folder"
  fi
done
