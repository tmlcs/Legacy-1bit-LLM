#!/bin/bash
# deploy.sh - Deployment script for Legacy-1bit LLM
# Usage: ./deploy.sh [install_dir]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VERSION="1.0.0"
PROJECT_NAME="legacy-1bit-llm"
DEFAULT_INSTALL_DIR="/usr/local"
INSTALL_DIR="${1:-$DEFAULT_INSTALL_DIR}"

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if running as root for system-wide install
check_permissions() {
    if [[ "$INSTALL_DIR" == "/usr/local" ]] && [[ $EUID -ne 0 ]]; then
        print_warning "Installing to $INSTALL_DIR requires root privileges"
        print_info "Run with sudo or choose a different install directory"
        print_info "Usage: sudo ./deploy.sh [install_dir]"
        exit 1
    fi
}

# Create directory structure
create_directories() {
    print_info "Creating directory structure..."
    
    mkdir -p "$INSTALL_DIR/bin"
    mkdir -p "$INSTALL_DIR/share/$PROJECT_NAME/config"
    mkdir -p "$INSTALL_DIR/share/$PROJECT_NAME/data"
    mkdir -p "$INSTALL_DIR/share/doc/$PROJECT_NAME"
    mkdir -p "$INSTALL_DIR/var/log/$PROJECT_NAME"
    
    print_info "Directories created"
}

# Install binaries
install_binaries() {
    print_info "Installing binaries..."
    
    # Check if binaries exist
    if [[ ! -f "legacy_llm_sse" ]]; then
        print_warning "Binaries not found. Building first..."
        ./build.sh all
    fi
    
    # Install training binaries
    cp legacy_llm_sse "$INSTALL_DIR/bin/" 2>/dev/null || cp legacy_llm_no_sse "$INSTALL_DIR/bin/"
    cp legacy_llm_no_sse "$INSTALL_DIR/bin/"
    
    # Install inference binaries
    cp inference_sse "$INSTALL_DIR/bin/" 2>/dev/null || cp inference_no_sse "$INSTALL_DIR/bin/"
    cp inference_no_sse "$INSTALL_DIR/bin/"
    
    # Make executable
    chmod +x "$INSTALL_DIR/bin/"legacy_llm*
    chmod +x "$INSTALL_DIR/bin/"inference*
    
    print_info "Binaries installed to $INSTALL_DIR/bin"
}

# Install configuration files
install_config() {
    print_info "Installing configuration files..."
    
    if [[ -f "config/example_config.txt" ]]; then
        cp config/example_config.txt "$INSTALL_DIR/share/$PROJECT_NAME/config/"
        print_info "Configuration template installed"
    fi
}

# Install documentation
install_docs() {
    print_info "Installing documentation..."
    
    # Copy main documentation files
    for doc in README.md PROJECT_PLAN.md ACTION_PLAN.md AGENTS.md; do
        if [[ -f "$doc" ]]; then
            cp "$doc" "$INSTALL_DIR/share/doc/$PROJECT_NAME/"
        fi
    done
    
    # Create man page (optional)
    mkdir -p "$INSTALL_DIR/share/man/man1"
    cat > "$INSTALL_DIR/share/man/man1/legacy_llm.1" << 'EOF'
.\" Man page for Legacy-1bit LLM
.TH LEGACY_LLM 1 "February 2026" "Version 1.0.0" "Legacy-1bit LLM"
.SH NAME
legacy_llm_sse, legacy_llm_no_sse \- Ternary Large Language Model for 2000-era hardware
.SH SYNOPSIS
.B legacy_llm_sse
.RI [ options ]
.br
.B inference_sse
.RI "'prompt'"
.SH DESCRIPTION
Train and run a simplified Large Language Model using ternary weights (-1, 0, 1)
optimized for hardware from the 2000 era.
.SH OPTIONS
.TP
.BI "-c, --config " file
Use configuration file
.TP
.BI "-m, --model " file
Model file path (default: llm_model.bin)
.TP
.B "-h, --help"
Show help message
.SH EXAMPLES
Train a new model:
.PP
.nf
.RS
legacy_llm_sse
.RE
.fi
.PP
Generate text:
.PP
.nf
.RS
inference_sse "Once upon a time"
.RE
.fi
.SH SEE ALSO
.BR inference_sse (1)
.SH AUTHOR
Legacy-1bit LLM Project
.SH LICENSE
Open Source
EOF
    
    print_info "Documentation installed"
}

# Create wrapper scripts
create_wrappers() {
    print_info "Creating wrapper scripts..."
    
    # Create a convenient training script
    cat > "$INSTALL_DIR/bin/llm-train" << EOF
#!/bin/bash
# Wrapper script for training

CONFIG_DIR="\$HOME/.config/$PROJECT_NAME"
LOG_DIR="\$HOME/.local/share/$PROJECT_NAME/logs"

# Create directories if needed
mkdir -p "\$CONFIG_DIR"
mkdir -p "\$LOG_DIR"

# Check if user has a config file
if [[ -f "\$CONFIG_DIR/config.txt" ]]; then
    echo "Using user configuration: \$CONFIG_DIR/config.txt"
fi

# Run training
cd "\$LOG_DIR"
exec legacy_llm_sse "\$@"
EOF
    
    # Create inference wrapper
    cat > "$INSTALL_DIR/bin/llm-generate" << 'EOF'
#!/bin/bash
# Wrapper script for text generation

if [[ $# -eq 0 ]]; then
    echo "Usage: llm-generate 'Your prompt here'"
    echo "       llm-generate -s temperature -t 0.8 'Prompt'"
    exit 1
fi

exec inference_sse "$@"
EOF
    
    chmod +x "$INSTALL_DIR/bin/llm-train"
    chmod +x "$INSTALL_DIR/bin/llm-generate"
    
    print_info "Wrapper scripts created"
}

# Create uninstall script
create_uninstall() {
    print_info "Creating uninstall script..."
    
    cat > "$INSTALL_DIR/bin/llm-uninstall" << EOF
#!/bin/bash
# Uninstall Legacy-1bit LLM

echo "Removing Legacy-1bit LLM..."
rm -f "$INSTALL_DIR/bin/legacy_llm*"
rm -f "$INSTALL_DIR/bin/inference*"
rm -f "$INSTALL_DIR/bin/llm-train"
rm -f "$INSTALL_DIR/bin/llm-generate"
rm -f "$INSTALL_DIR/bin/llm-uninstall"
rm -rf "$INSTALL_DIR/share/$PROJECT_NAME"
rm -rf "$INSTALL_DIR/share/doc/$PROJECT_NAME"
rm -f "$INSTALL_DIR/share/man/man1/legacy_llm.1"
echo "Legacy-1bit LLM has been uninstalled"
EOF
    
    chmod +x "$INSTALL_DIR/bin/llm-uninstall"
    print_info "Uninstall script created"
}

# Print installation summary
print_summary() {
    print_header ""
    print_header "========================================"
    print_header "  Deployment Complete!"
    print_header "========================================"
    print_header ""
    print_info "Installed binaries:"
    echo "  - legacy_llm_sse"
    echo "  - legacy_llm_no_sse"
    echo "  - inference_sse"
    echo "  - inference_no_sse"
    echo "  - llm-train (wrapper)"
    echo "  - llm-generate (wrapper)"
    print_header ""
    print_info "Installation directory: $INSTALL_DIR"
    print_info "Configuration: $INSTALL_DIR/share/$PROJECT_NAME/config/"
    print_info "Documentation: $INSTALL_DIR/share/doc/$PROJECT_NAME/"
    print_info "Logs: $INSTALL_DIR/var/log/$PROJECT_NAME/"
    print_header ""
    print_info "Quick start:"
    echo "  1. Train: llm-train"
    echo "  2. Generate: llm-generate 'Your prompt'"
    echo "  3. Uninstall: llm-uninstall"
    print_header ""
    print_info "For more options, see: man legacy_llm"
    print_header ""
}

# Main deployment
main() {
    print_header ""
    print_header "Legacy-1bit LLM Deployment Script"
    print_header "================================="
    print_header ""
    
    print_info "Version: $VERSION"
    print_info "Install directory: $INSTALL_DIR"
    
    # Check permissions
    check_permissions
    
    # Build if needed
    if [[ ! -f "legacy_llm_sse" ]] && [[ ! -f "legacy_llm_no_sse" ]]; then
        print_info "Building project..."
        ./build.sh all
    fi
    
    # Install
    create_directories
    install_binaries
    install_config
    install_docs
    create_wrappers
    create_uninstall
    
    # Summary
    print_summary
}

# Run
main "$@"
