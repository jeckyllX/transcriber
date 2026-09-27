#!/usr/bin/env bash
# ==============================================================================
# Automated Setup Script for Audio Transcriber Platform
# Modular, Portable, Zero Hardcoding, Cross-Architecture (x86_64 / aarch64)
# ==============================================================================

set -euo pipefail

# Determine project root dynamically
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${PROJECT_DIR}/bin"
MODELS_DIR="${PROJECT_DIR}/models"
UPLOADS_DIR="${PROJECT_DIR}/uploads"
VENV_DIR="${PROJECT_DIR}/.venv"

# Create standard directories
mkdir -p "${BIN_DIR}" "${MODELS_DIR}" "${UPLOADS_DIR}"

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Multi-architecture and OS detection (x86_64, aarch64, armv7l, i686, riscv64)
RAW_ARCH="$(uname -m)"
RAW_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

case "${RAW_ARCH}" in
    x86_64|amd64)
        ARCH_WHISPER="ubuntu-x64"
        ARCH_OLLAMA="amd64"
        ARCH_STATIC_FFMPEG="amd64"
        ;;
    aarch64|arm64)
        ARCH_WHISPER="ubuntu-arm64"
        ARCH_OLLAMA="arm64"
        ARCH_STATIC_FFMPEG="arm64"
        ;;
    armv7l|armhf)
        ARCH_WHISPER="unknown"
        ARCH_OLLAMA="arm"
        ARCH_STATIC_FFMPEG="armhf"
        ;;
    i686|i386)
        ARCH_WHISPER="unknown"
        ARCH_OLLAMA="i386"
        ARCH_STATIC_FFMPEG="i686"
        ;;
    riscv64)
        ARCH_WHISPER="unknown"
        ARCH_OLLAMA="unknown"
        ARCH_STATIC_FFMPEG="unknown"
        warn "RISC-V 64 detected: native source compilation will be used."
        ;;
    *)
        ARCH_WHISPER="unknown"
        ARCH_OLLAMA="unknown"
        ARCH_STATIC_FFMPEG="unknown"
        warn "Architecture ${RAW_ARCH} detected: source compilation will be used."
        ;;
esac

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}    Audio Transcriber - Automated Environment Setup    ${NC}"
echo -e "${BLUE}======================================================${NC}"
info "Project Directory: ${PROJECT_DIR}"
info "Architecture: ${RAW_ARCH}"

# ------------------------------------------------------------------------------
# 1. FFmpeg Verification
# ------------------------------------------------------------------------------
# Package manager abstraction supporting apt, dnf, yum, pacman, zypper, apk, and brew
install_system_package() {
    local PKG="$1"
    local ELEVATE=""

    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            ELEVATE="sudo"
        else
            return 1
        fi
    fi

    if command -v apt-get >/dev/null 2>&1; then
        ${ELEVATE} apt-get update -qq && ${ELEVATE} apt-get install -y "${PKG}"
    elif command -v dnf >/dev/null 2>&1; then
        ${ELEVATE} dnf install -y "${PKG}"
    elif command -v yum >/dev/null 2>&1; then
        ${ELEVATE} yum install -y "${PKG}"
    elif command -v pacman >/dev/null 2>&1; then
        ${ELEVATE} pacman -Sy --noconfirm "${PKG}"
    elif command -v zypper >/dev/null 2>&1; then
        ${ELEVATE} zypper --non-interactive install "${PKG}"
    elif command -v apk >/dev/null 2>&1; then
        ${ELEVATE} apk add "${PKG}"
    elif command -v brew >/dev/null 2>&1; then
        brew install "${PKG}"
    else
        return 1
    fi
}

info "Checking FFmpeg..."
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_PATH="$(command -v ffmpeg)"
    success "FFmpeg found at: ${FFMPEG_PATH}"
else
    warn "FFmpeg is not in PATH. Attempting package manager installation..."
    if install_system_package "ffmpeg"; then
        success "FFmpeg installed via system package manager."
    else
        warn "Package manager unavailable or unprivileged. Attempting architecture-specific static download..."
        if [ "${ARCH_STATIC_FFMPEG}" != "unknown" ]; then
            FFMPEG_STATIC_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${ARCH_STATIC_FFMPEG}-static.tar.xz"
            if curl -f -sSL "${FFMPEG_STATIC_URL}" | tar -xJ --strip-components=1 -C "${BIN_DIR}"; then
                success "Static FFmpeg (${ARCH_STATIC_FFMPEG}) extracted to ${BIN_DIR}."
            else
                warn "Static download failed for ${ARCH_STATIC_FFMPEG}. Ensure FFmpeg is installed manually."
            fi
        else
            warn "No static binary available for ${RAW_ARCH}. Please install FFmpeg via your system package manager."
        fi
    fi
fi

# ------------------------------------------------------------------------------
# 2. Whisper Binary (whisper.cpp) Setup & Verification
# ------------------------------------------------------------------------------
info "Checking Whisper standalone binary (whisper.cpp)..."
WHISPER_CLI_BIN="${BIN_DIR}/whisper-cli"
WHISPER_NEEDS_INSTALL=true

if [ -x "${WHISPER_CLI_BIN}" ]; then
    info "Found existing whisper-cli binary in ${BIN_DIR}. Testing execution..."
    if LD_LIBRARY_PATH="${BIN_DIR}:${LD_LIBRARY_PATH:-}" "${WHISPER_CLI_BIN}" --help >/dev/null 2>&1; then
        success "Existing whisper-cli is operational."
        WHISPER_NEEDS_INSTALL=false
    else
        warn "Existing whisper-cli failed verification test. Re-installing..."
    fi
fi

if [ "${WHISPER_NEEDS_INSTALL}" = true ]; then
    DOWNLOAD_SUCCESS=false
    if [ "${ARCH_WHISPER}" != "unknown" ]; then
        info "Querying latest binary release for ggml-org/whisper.cpp on GitHub..."
        ARCHIVE_NAME="whisper-bin-${ARCH_WHISPER}.tar.gz"
        DOWNLOAD_URL=$(curl -sL https://api.github.com/repos/ggml-org/whisper.cpp/releases | grep -o "https://github.com/ggml-org/whisper.cpp/releases/download/[^\"/]*/${ARCHIVE_NAME}" | head -n 1 || echo "")
        if [ -z "${DOWNLOAD_URL}" ]; then
            DOWNLOAD_URL="https://github.com/ggml-org/whisper.cpp/releases/download/b5130/${ARCHIVE_NAME}"
        fi

        info "Downloading precompiled binary from: ${DOWNLOAD_URL}"
        TMP_ARCHIVE="${BIN_DIR}/${ARCHIVE_NAME}"
        if curl -f -sSL -o "${TMP_ARCHIVE}" "${DOWNLOAD_URL}"; then
            tar -xzf "${TMP_ARCHIVE}" -C "${BIN_DIR}" --strip-components=1
            rm -f "${TMP_ARCHIVE}"

            if [ -f "${BIN_DIR}/whisper-cli" ]; then
                mv -f "${BIN_DIR}/whisper-cli" "${BIN_DIR}/whisper-cli-bin"
            elif [ -f "${BIN_DIR}/main" ]; then
                mv -f "${BIN_DIR}/main" "${BIN_DIR}/whisper-cli-bin"
            fi

            # Create runner script to ensure LD_LIBRARY_PATH resolves local shared libraries
            cat << 'WRAPPER' > "${WHISPER_CLI_BIN}"
#!/usr/bin/env bash
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SELF_DIR}:${LD_LIBRARY_PATH:-}"
if [ -x "${SELF_DIR}/whisper-cli-bin" ]; then
    exec "${SELF_DIR}/whisper-cli-bin" "$@"
elif [ -x "${SELF_DIR}/main" ]; then
    exec "${SELF_DIR}/main" "$@"
else
    echo "whisper binary not found in ${SELF_DIR}" >&2
    exit 1
fi
WRAPPER
            chmod +x "${WHISPER_CLI_BIN}"

            if "${WHISPER_CLI_BIN}" --help >/dev/null 2>&1; then
                success "Precompiled whisper.cpp binary verified successfully!"
                DOWNLOAD_SUCCESS=true
            else
                warn "Precompiled binary did not execute cleanly. Falling back to source compilation."
            fi
        else
            warn "Could not download precompiled binary. Falling back to source compilation."
        fi
    fi

    if [ "${DOWNLOAD_SUCCESS}" = false ]; then
        info "Compiling whisper.cpp from source (CMake/Make) for ${RAW_ARCH}..."
        if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
            info "Installing build prerequisites (cmake, build-essential/gcc)..."
            install_system_package "cmake" || true
            install_system_package "build-essential" 2>/dev/null || install_system_package "gcc-c++" 2>/dev/null || install_system_package "base-devel" 2>/dev/null || true
        fi
        BUILD_DIR="${PROJECT_DIR}/.whisper_build"
        rm -rf "${BUILD_DIR}"
        git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "${BUILD_DIR}"
        cmake -S "${BUILD_DIR}" -B "${BUILD_DIR}/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "${BUILD_DIR}/build" -j"$(nproc)" --config Release

        if [ -f "${BUILD_DIR}/build/bin/whisper-cli" ]; then
            cp "${BUILD_DIR}/build/bin/whisper-cli" "${BIN_DIR}/whisper-cli-bin"
        elif [ -f "${BUILD_DIR}/build/bin/main" ]; then
            cp "${BUILD_DIR}/build/bin/main" "${BIN_DIR}/whisper-cli-bin"
        fi
        cp -P "${BUILD_DIR}/build/bin/"*.so* "${BIN_DIR}/" 2>/dev/null || true

        cat << 'WRAPPER' > "${WHISPER_CLI_BIN}"
#!/usr/bin/env bash
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SELF_DIR}:${LD_LIBRARY_PATH:-}"
if [ -x "${SELF_DIR}/whisper-cli-bin" ]; then
    exec "${SELF_DIR}/whisper-cli-bin" "$@"
elif [ -x "${SELF_DIR}/main" ]; then
    exec "${SELF_DIR}/main" "$@"
else
    echo "whisper binary not found in ${SELF_DIR}" >&2
    exit 1
fi
WRAPPER
        chmod +x "${WHISPER_CLI_BIN}"
        success "Compiled whisper-cli installed to ${BIN_DIR}."
        rm -rf "${BUILD_DIR}"
    fi
fi

# ------------------------------------------------------------------------------
# 3. Download Default Whisper GGML Model with SHA256 Verification
# ------------------------------------------------------------------------------
DEFAULT_MODEL_NAME="ggml-base.bin"
DEFAULT_MODEL_FILE="${MODELS_DIR}/${DEFAULT_MODEL_NAME}"
BASE_MODEL_SHA256="60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"

if [ -f "${DEFAULT_MODEL_FILE}" ]; then
    info "Verifying checksum for existing model: ${DEFAULT_MODEL_NAME}..."
    ACTUAL_SHA="$(sha256sum "${DEFAULT_MODEL_FILE}" | awk '{print $1}')"
    if [ "${ACTUAL_SHA}" = "${BASE_MODEL_SHA256}" ]; then
        success "Model ${DEFAULT_MODEL_NAME} verified (SHA256 valid)."
    else
        warn "Model checksum mismatch. Re-downloading..."
        rm -f "${DEFAULT_MODEL_FILE}"
    fi
fi

if [ ! -f "${DEFAULT_MODEL_FILE}" ]; then
    MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${DEFAULT_MODEL_NAME}"
    info "Downloading Whisper model from: ${MODEL_URL}"
    curl -f -L -# -o "${DEFAULT_MODEL_FILE}" "${MODEL_URL}"
    ACTUAL_SHA="$(sha256sum "${DEFAULT_MODEL_FILE}" | awk '{print $1}')"
    if [ "${ACTUAL_SHA}" = "${BASE_MODEL_SHA256}" ]; then
        success "Whisper model downloaded and SHA256 checksum verified."
    else
        warn "Downloaded model SHA256 was ${ACTUAL_SHA}, expected ${BASE_MODEL_SHA256}."
    fi
fi

# ------------------------------------------------------------------------------
# 4. Speaker Diarization Models (Sherpa-ONNX Pyannote + 3D-Speaker)
# ------------------------------------------------------------------------------
info "Checking speaker diarization ONNX models..."
SEG_MODEL_DIR="${MODELS_DIR}/sherpa-onnx-pyannote-segmentation-3-0"
EMB_MODEL_FILE="${MODELS_DIR}/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

if [ ! -f "${SEG_MODEL_DIR}/model.onnx" ] && [ ! -f "${SEG_MODEL_DIR}/model.int8.onnx" ]; then
    info "Downloading Sherpa-ONNX segmentation model (Pyannote 3.0)..."
    SEG_TAR="${MODELS_DIR}/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    curl -f -L -# -o "${SEG_TAR}" "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    tar -xjf "${SEG_TAR}" -C "${MODELS_DIR}"
    rm -f "${SEG_TAR}"
    success "Speaker segmentation model extracted."
else
    success "Speaker segmentation model found."
fi

if [ ! -f "${EMB_MODEL_FILE}" ]; then
    info "Downloading Sherpa-ONNX speaker embedding model (3D-Speaker)..."
    curl -f -L -# -o "${EMB_MODEL_FILE}" "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    success "Speaker embedding model downloaded."
else
    success "Speaker embedding model found."
fi

# ------------------------------------------------------------------------------
# 5. Ollama Status Check
# ------------------------------------------------------------------------------
info "Checking Ollama connectivity..."
if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    success "Ollama is active and responding on http://127.0.0.1:11434"
elif command -v ollama >/dev/null 2>&1; then
    success "Ollama binary found in PATH. You can start it anytime via 'ollama serve'."
else
    info "Ollama is not running locally. You can install it, run 'ollama serve', or configure a remote host in the web UI."
fi

# ------------------------------------------------------------------------------
# 6. Python Virtual Environment Setup
# ------------------------------------------------------------------------------
info "Setting up Python virtual environment in ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel --quiet
"${VENV_DIR}/bin/pip" install -r "${PROJECT_DIR}/requirements.txt" --quiet
success "Python virtual environment configured with all dependencies."

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}   Setup Complete! Ready to launch the platform.       ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Start the application with: ${YELLOW}bash run.sh${NC}"
