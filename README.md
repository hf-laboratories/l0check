# l0check — Intel Level Zero Hardware Diagnostics & Validation Tool

`l0check` is a fast, standalone command-line diagnostic tool for inspecting Intel® GPU architectures, driver capabilities, memory topologies, and validating compute kernel execution via the Intel oneAPI Level Zero runtime.

---

## 🚀 Features

- **Driver & Device Enumeration**: Query driver versions, API versions, device IDs, PCI bus addresses, and execution unit configurations.
- **Topology Inspection**: Analyze memory subsystems, sub-device / compute slice topologies, and cache hierarchy.
- **Compute Kernel Verification**: Run comprehensive diagnostic smoke tests and compute kernel pipelines against the Level Zero driver.
- **Standalone & Lightweight**: Self-contained executable with zero native setup dependencies.

---

## 📦 Quick Start

### Run Diagnostic Scan
```shell
l0check
```

### Run Full Hardware & Kernel Diagnostics
```shell
l0check --test
```

---

## 🛠️ Build from Source

Requirements:
- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- Intel Graphics Driver with Level Zero runtime support (`ze_loader.dll`)

```shell
git clone https://github.com/hf-laboratories/l0check.git
cd l0check
dotnet run --project src/LevelZeroCheck.csproj
```

### Publish Self-Contained Binary
```shell
dotnet publish src/LevelZeroCheck.csproj -c Release -r win-x64 --self-contained -p:PublishSingleFile=true -o dist/
```

---

## 📄 License

Apache-2.0. See [LICENSE](LICENSE) for details.
