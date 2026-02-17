# Deployment Guide for Legacy-1bit LLM

**Version:** 1.0.0  
**Date:** February 2026  
**Status:** Production Ready

---

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Clone or extract the project
cd legacy-1bit-llm

# Run deployment script
sudo ./deploy.sh

# Or install to custom location
./deploy.sh /opt/legacy-llm
```

### Option 2: Manual Installation

```bash
# Build the project
make clean
make all

# Copy binaries to PATH
sudo cp legacy_llm_sse inference_sse /usr/local/bin/

# Create necessary directories
mkdir -p ~/.config/legacy-1bit-llm
mkdir -p ~/.local/share/legacy-1bit-llm/logs
```

---

## 📋 System Requirements

### Minimum Requirements
- **OS:** Linux (x86_64), macOS, or Windows with WSL
- **CPU:** x86 processor (SSE/SSE2 support optional)
- **RAM:** 64 MB minimum, 256 MB recommended
- **Storage:** 10 MB for installation, 100 MB for models and logs
- **Compiler:** GCC 4.8+ or Clang 3.4+ with C99 support

### Dependencies
- `gcc` or `clang` (for building from source)
- `make` (for building)
- `math` library (libm, usually included)

### Optional Dependencies
- **SSE/SSE2:** For 2x performance improvement
- **Valgrind:** For memory leak testing
- **Git:** For version control

---

## 🔧 Building from Source

### Prerequisites

Ensure you have the required tools:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential gcc make

# CentOS/RHEL/Fedora
sudo yum groupinstall "Development Tools"
# or
sudo dnf groupinstall "Development Tools"

# macOS
xcode-select --install

# Verify installation
gcc --version
make --version
```

### Build Instructions

```bash
# 1. Extract or clone the project
cd legacy-1bit-llm

# 2. Run the build script
./build.sh all

# Or use make directly
make clean
make legacy_llm_sse inference_sse

# 3. Verify binaries were created
ls -la legacy_llm* inference*
```

### Build Options

```bash
# Build all targets (SSE + Non-SSE)
make all

# Build only SSE-optimized versions
make legacy_llm_sse inference_sse

# Build only Non-SSE versions (for older CPUs)
make legacy_llm_no_sse inference_no_sse

# Build test runners
make test_runner_sse test_runner_no_sse

# Run all tests
make test

# Clean build artifacts
make clean
```

---

## 📦 Installation Methods

### Method 1: System-wide Installation

```bash
# Install to /usr/local (requires root)
sudo ./deploy.sh

# Verify installation
which legacy_llm_sse
which inference_sse

# Check version
legacy_llm_sse --help
```

### Method 2: User Installation

```bash
# Install to user directory
./deploy.sh $HOME/.local

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Method 3: Docker Deployment

```dockerfile
# Dockerfile
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make

WORKDIR /app
COPY . .

RUN make clean && make all

CMD ["./legacy_llm_sse"]
```

Build and run:
```bash
docker build -t legacy-llm .
docker run -v $(pwd)/data:/app/data legacy-llm
```

---

## ⚙️ Configuration

### Configuration File

Create a configuration file at `~/.config/legacy-1bit-llm/config.txt`:

```bash
# Copy example configuration
mkdir -p ~/.config/legacy-1bit-llm
cp /usr/local/share/legacy-1bit-llm/config/example_config.txt \
   ~/.config/legacy-1bit-llm/config.txt

# Edit configuration
nano ~/.config/legacy-1bit-llm/config.txt
```

### Environment Variables

```bash
# Set default model path
export LLM_MODEL_PATH="/path/to/model.bin"

# Set log directory
export LLM_LOG_DIR="/path/to/logs"

# Set configuration file
export LLM_CONFIG="/path/to/config.txt"
```

### Using Different Configurations

```bash
# Train with custom config
legacy_llm_sse --config my_config.txt

# Generate with specific model
inference_sse --model my_model.bin "Prompt"
```

---

## 🧪 Testing Installation

### Run Built-in Tests

```bash
# Run all tests
make test

# Or run specific test runner
./test_runner_sse

# Expected output: All 31 tests should pass
```

### Test Training

```bash
# Quick training test (1 epoch)
./legacy_llm_sse --epochs 1

# Check logs were created
ls logs/training_*.json
```

### Test Inference

```bash
# Generate text
./inference_sse "Once upon a time"

# Should output generated text
```

---

## 📊 Production Deployment

### Performance Optimization

1. **Use SSE Version:** Always use `legacy_llm_sse` on modern CPUs (2x faster)

2. **CPU Affinity:** Pin process to specific CPU cores
   ```bash
   taskset -c 0,1,2,3 ./legacy_llm_sse
   ```

3. **Memory Optimization:** Use streaming for large datasets
   ```bash
   # Enable large file support
   echo "chunk_size=16384" >> config.txt
   ```

4. **Batch Processing:** For production inference
   ```bash
   # Process multiple prompts
   cat prompts.txt | xargs -I {} inference_sse "{}"
   ```

### Monitoring

```bash
# Monitor resource usage
htop

# Check logs in real-time
tail -f logs/training_*.json

# Performance profiling
make perf
```

### Backup Strategy

```bash
# Backup model checkpoints
cp llm_model.bin llm_model_backup_$(date +%Y%m%d).bin

# Backup configuration
tar czf config_backup.tar.gz config/

# Backup logs
tar czf logs_backup.tar.gz logs/
```

---

## 🐛 Troubleshooting

### Build Issues

#### Issue: `gcc: error: unrecognized command line option '-std=c99'`
**Solution:** Update GCC to version 4.8 or later
```bash
# Check version
gcc --version

# Update on Ubuntu
sudo apt-get install gcc-8
```

#### Issue: `undefined reference to 'sqrtf'`
**Solution:** Link math library
```bash
# Already handled in Makefile, but if building manually:
gcc -o output file.c -lm
```

### Runtime Issues

#### Issue: `Illegal instruction (core dumped)`
**Solution:** CPU doesn't support SSE, use Non-SSE version
```bash
./legacy_llm_no_sse  # Instead of legacy_llm_sse
```

#### Issue: `Cannot load model from llm_model.bin`
**Solution:** Train model first or provide correct path
```bash
# Train first
./legacy_llm_sse

# Or specify path
./inference_sse --model /path/to/model.bin "prompt"
```

#### Issue: `Out of memory`
**Solution:** Reduce model dimensions or batch size
```bash
# Edit config.txt
model_dim=128        # Instead of 256
batch_size=4         # Instead of 8
```

### Performance Issues

#### Issue: Training is too slow
**Solutions:**
1. Use SSE version: `legacy_llm_sse`
2. Reduce model dimensions
3. Enable SSE4.1 if available: `make legacy_llm_sse`
4. Use smaller dataset for testing

#### Issue: High memory usage
**Solutions:**
1. Use dataset streaming (enabled by default for large files)
2. Reduce `chunk_size` in config
3. Lower `model_dim` and `num_transformer_blocks`

---

## 🔒 Security Considerations

1. **File Permissions:**
   ```bash
   # Set secure permissions on model files
   chmod 600 llm_model.bin
   
   # Set permissions on config
   chmod 644 config.txt
   ```

2. **Input Validation:**
   - Always validate input prompts
   - Sanitize file paths in configuration
   - Use absolute paths when possible

3. **Resource Limits:**
   ```bash
   # Limit CPU usage
   cpulimit -l 50 ./legacy_llm_sse
   
   # Limit memory
   ulimit -v 524288  # 512 MB
   ```

---

## 🔄 Updating

### Update to New Version

```bash
# 1. Backup current installation
llm-uninstall  # Creates backup automatically
# or
sudo cp -r /usr/local/share/legacy-1bit-llm ~/llm-backup

# 2. Download new version
cd /tmp
wget https://github.com/user/legacy-1bit-llm/releases/latest/download/legacy-1bit-llm.tar.gz
tar xzf legacy-1bit-llm.tar.gz
cd legacy-1bit-llm

# 3. Deploy new version
sudo ./deploy.sh

# 4. Restore configuration
cp ~/llm-backup/config/* ~/.config/legacy-1bit-llm/
```

---

## 📞 Support

### Getting Help

1. **Check Documentation:**
   - README.md - General overview
   - AGENTS.md - Developer guide
   - PROJECT_PLAN.md - Architecture details

2. **Run Tests:**
   ```bash
   make test
   ```

3. **Enable Debug Mode:**
   ```bash
   # Build with debug symbols
   make clean
   CFLAGS="-g -O0" make all
   
   # Run with debugger
   gdb ./legacy_llm_sse
   ```

### Reporting Issues

When reporting issues, include:
- OS and version
- GCC version (`gcc --version`)
- CPU information (`cat /proc/cpuinfo | grep "model name"`)
- Error messages (full output)
- Configuration file (if modified)

---

## 🎉 Next Steps

After successful deployment:

1. **Train Your First Model:**
   ```bash
   llm-train
   ```

2. **Generate Text:**
   ```bash
   llm-generate "Once upon a time"
   ```

3. **Experiment with Configurations:**
   ```bash
   # Try different hyperparameters
   nano ~/.config/legacy-1bit-llm/config.txt
   
   # Train with new config
   llm-train
   ```

4. **Monitor Performance:**
   ```bash
   # Check logs
   ls ~/.local/share/legacy-1bit-llm/logs/
   
   # View metrics
   cat logs/training_*.json | grep "loss"
   ```

---

## 📄 License

This project is open source. See LICENSE file for details.

---

**Congratulations! Your Legacy-1bit LLM is now deployed and ready to use.** 🚀
