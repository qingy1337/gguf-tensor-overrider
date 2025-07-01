#!/bin/bash

# Check if user has node installed and if it's version >=22.6.0
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js version >=22.6.0."
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'.' -f1 | sed 's/v//')
if [ "$NODE_VERSION" -lt 22 ] || { [ "$NODE_VERSION" -eq 22 ] && [ "$(node -v | cut -d'.' -f2)" -lt 6 ]; }; then
    echo "Node.js version must be >=22.6.0. Please update Node.js."
    exit 1
fi


# Create installation directory
INSTALL_DIR="/usr/local/lib/gguf-tensor-overrider"
BIN_FILE="/usr/local/bin/gguf-tensor-overrider"

# Remove existing installation directory if it exists
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing existing installation..."
    sudo rm -rf "$INSTALL_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to remove existing installation directory. Check permissions."
        exit 1
    fi
fi

# Create new installation directory
sudo mkdir -p "$INSTALL_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to create directory $INSTALL_DIR. Please check your permissions."
    exit 1
fi

# Copy files to installation directory
git clone https://github.com/k-koehler/gguf-tensor-overrider "$INSTALL_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to clone repository to $INSTALL_DIR. Please check your internet connection or repository URL."
    exit 1
fi

# Run npm install in the installation directory
cd "$INSTALL_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to change directory to $INSTALL_DIR. Please check your permissions."
    exit 1
fi  
npm install
if [ $? -ne 0 ]; then
    echo "npm install failed. Please check the output for errors."
    exit 1
fi

# Remove existing binary file if it exists
if [ -f "$BIN_FILE" ]; then
    sudo rm "$BIN_FILE"
    if [ $? -ne 0 ]; then
        echo "Failed to remove existing binary. Check permissions."
        exit 1
    fi
fi

# Create binary file
echo "node --experimental-strip-types --disable-warning=ExperimentalWarning $INSTALL_DIR/index.js" | sudo tee "$BIN_FILE" > /dev/null
if [ $? -ne 0 ]; then
    echo "Failed to create binary file $BIN_FILE. Please check your permissions."
    exit 1
fi
sudo chmod +x "$BIN_FILE"
if [ $? -ne 0 ]; then
    echo "Failed to make $BIN_FILE executable. Please check your permissions."
    exit 1
fi

# Print success message
echo "gguf-tensor-overrider installed successfully!"