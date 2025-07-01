#!/bin/bash

# check if user has node installed and if it's version >=22.6.0
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js version >=22.6.0."
    exit 1
fi
NODE_VERSION=$(node -v | cut -d'.' -f1 | sed 's/v//')
if [ "$NODE_VERSION" -lt 22 ] || { [ "$NODE_VERSION" -eq 22 ] && [ "$(node -v | cut -d'.' -f2)" -lt 6 ]; }; then
    echo "Node.js version must be >=22.6.0. Please update Node.js."
    exit 1
fi

# create a folder in /usr/local/lib/gguf-tensor-overrider
INSTALL_DIR="/usr/local/lib/gguf-tensor-overrider"
if [ ! -d "$INSTALL_DIR" ]; then
    sudo mkdir -p "$INSTALL_DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create directory $INSTALL_DIR. Please check your permissions."
        exit 1
    fi
fi

# copy the current directory contents to the install directory
cp -r ./* "$INSTALL_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to copy files to $INSTALL_DIR. Please check your permissions."
    exit 1
fi

# run npm install in the install directory
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

# create a file gguf-tensor-overrider in /usr/local/bin
# with contents: node /usr/local/lib/gguf-tensor-overrider/index.js
BIN_FILE="/usr/local/bin/gguf-tensor-overrider"
if [ ! -f "$BIN_FILE" ]; then
    echo "node $INSTALL_DIR/index.js" | sudo tee "$BIN_FILE" > /dev/null
    if [ $? -ne 0 ]; then
        echo "Failed to create binary file $BIN_FILE. Please check your permissions."
        exit 1
    fi
    sudo chmod +x "$BIN_FILE"
    if [ $? -ne 0 ]; then
        echo "Failed to make $BIN_FILE executable. Please check your permissions."
        exit 1
    fi
else
    echo "Binary file $BIN_FILE already exists. Please remove it before running the install script again."
    exit 1
fi  

# print success message
echo "gguf-tensor-overrider installed successfully!"