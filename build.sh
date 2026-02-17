#!/bin/bash
# build.sh - Build script for Legacy-1bit LLM
# Usage: ./build.sh [target]
# Targets: all, sse, no-sse, inference, test, clean

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command_exists gcc; then
        print_error "gcc is required but not installed"
        exit 1
    fi
    
    if ! command_exists make; then
        print_error "make is required but not installed"
        exit 1
    fi
    
    # Check gcc version (need C99 support)
    GCC_VERSION=$(gcc --version | head -n1 | grep -oP '\d+\.\d+' | head -1)
    print_info "Found gcc version: $GCC_VERSION"
    
    # Check for SSE support
    if grep -q "sse" /proc/cpuinfo 2>/dev/null || [[ "$OSTYPE" == "darwin"* ]]; then
        print_info "SSE support detected"
        SSE_SUPPORTED=1
    else
        print_warning "SSE support not detected, will build non-SSE version only"
        SSE_SUPPORTED=0
    fi
}

# Build function
build_target() {
    local target=$1
    print_info "Building target: $target"
    
    case $target in
        "all")
            make clean
            make legacy_llm_no_sse legacy_llm_sse inference_no_sse inference_sse
            print_info "All targets built successfully"
            ;;
        "sse")
            make clean
            make legacy_llm_sse inference_sse
            print_info "SSE targets built successfully"
            ;;
        "no-sse")
            make clean
            make legacy_llm_no_sse inference_no_sse
            print_info "Non-SSE targets built successfully"
            ;;
        "inference")
            make inference_sse inference_no_sse
            print_info "Inference targets built successfully"
            ;;
        "test")
            make test_runner_no_sse test_runner_sse
            print_info "Test runners built successfully"
            ;;
        "clean")
            make clean
            print_info "Clean completed"
            ;;
        *)
            print_error "Unknown target: $target"
            print_info "Valid targets: all, sse, no-sse, inference, test, clean"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    local target="${1:-all}"
    
    print_info "Legacy-1bit LLM Build Script"
    print_info "============================"
    
    # Check dependencies
    check_dependencies
    
    # Build
    build_target "$target"
    
    # Verify binaries exist
    print_info "Verifying build artifacts..."
    
    if [[ -f "legacy_llm_no_sse" ]]; then
        print_info "✓ legacy_llm_no_sse built"
    fi
    
    if [[ -f "legacy_llm_sse" ]]; then
        print_info "✓ legacy_llm_sse built"
    fi
    
    if [[ -f "inference_no_sse" ]]; then
        print_info "✓ inference_no_sse built"
    fi
    
    if [[ -f "inference_sse" ]]; then
        print_info "✓ inference_sse built"
    fi
    
    print_info ""
    print_info "Build completed successfully!"
    print_info ""
    print_info "Next steps:"
    print_info "  1. Run tests: ./test_runner_sse"
    print_info "  2. Train model: ./legacy_llm_sse"
    print_info "  3. Generate text: ./inference_sse 'Your prompt'"
}

# Run main
main "$@"
