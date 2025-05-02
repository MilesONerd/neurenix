# Neurenix Installation Guide

This guide provides installation instructions for both **Phynexus** and **Neurenix**.  
Phynexus is the core engine of Neurenix, written in Rust. Neurenix is the main high-level AI framework built on top of Phynexus.

---

## Installing Phynexus (Rust)

Phynexus can be used independently in Rust projects. To install and start using it, follow the steps below.

### Step 1: Install Rust

If you haven't installed Rust yet, visit the official Rust website:

[https://www.rust-lang.org/](https://www.rust-lang.org/)

Or install it directly via terminal using `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the prompts to complete the installation.

### Step 2: Add Phynexus to Your Project

Once Rust is installed, you can add Phynexus to your Rust project with the following command:

```bash
cargo add phynexus
```

This will include the latest version of the Phynexus engine via `crates.io`.

---

## Installing Neurenix (Python)

Neurenix can be installed using either `pip` or `conda`, depending on your preferred Python environment.

### Option 1: Install via pip (Official and Primary Method)

To install the latest release of Neurenix from PyPI, use the following command:

```bash
pip install neurenix
```

This is the recommended way to install Neurenix for most users.

### Option 2: Install via Conda (Anaconda Distribution)

If you use Anaconda or Miniconda, you can install Neurenix using:

```bash
conda install neurenix::neurenix
```

This command installs Neurenix from the `neurenix` channel if available in your Conda environment.
