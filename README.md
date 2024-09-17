# Zerograd: A Deep Learning Framework

This repository contains instructions for setting up **Zerograd**, a deep learning framework. Below, you will find the necessary steps to set up dependencies such as the BLIS library, and other important information to get started.

## Requirements

Before proceeding, ensure the following requirements are met:

#### Step 1: Build and Install BLIS

Configure, build, and install BLIS. By default, BLIS will auto-detect the architecture and optimize accordingly. Optionally, you can specify an architecture for better performance during the configuration process.

```bash
./configure auto
make -j4    # Use -j4 or the number of cores you want to use for compilation
sudo make install
```

#### Step 2: Set the Library Path

After installing BLIS, you need to add the BLIS shared library path to the `LD_LIBRARY_PATH` environment variable so that it can be found by the dynamic linker.

#### Step 3: Run the Test Program

After compiling, run the test program in test dir. If everything is set up correctly, you should see the output of the program.


```bash
gcc test_blis.c -lblis -lm -o test_blis
```

### 2. Other Requirements

- GCC for compiling Zerograd.
- Make for building the project.

