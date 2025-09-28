# Neural Networks from Scratch in C++ and Real Dataset Testing

## Project Overview

This project involves developing neural networks from scratch in C++ and evaluating their performance on various datasets. It is divided into two main sections: 

1. **Neural Networks from Scratch in C++**: Implementation of various models including classification, binary logistic regression, and regression networks. Models are evaluated using synthetic datasets such as spiral, vertical, and sine data.

2. **A Real Dataset**: Application of the classification network to the Fashion MNIST dataset to demonstrate handling and analysis of real-world data.

## Directory Structure

### Neural Networks from Scratch in C++

#### Overview

This directory contains implementations for:

- **Classification Neural Network**:
  - **Activation Functions**: ReLU for hidden layers, Softmax for the output layer.
  - **Loss Function**: Categorical Cross-Entropy.

- **Binary Logistic Regression Neural Network**:
  - **Activation Functions**: ReLU for hidden layers, Sigmoid for the output layer.
  - **Loss Function**: Binary Cross-Entropy.

- **Regression Neural Network**:
  - **Activation Functions**: ReLU for hidden layers, Linear activation for the output layer.
  - **Loss Functions**: Mean Squared Error (MSE) or Mean Absolute Error (MAE).

#### Optimizers

- Stochastic Gradient Descent (SGD) with Momentum
- RMSProp
- AdaGrad
- Adam

Evaluations are performed using generated datasets, including spiral, vertical, and sine data, and accuracy is calculated with distinct techniques for each neural network.

#### Setup and Compilation

1. **Place the `.vscode` Directory:**
   - Ensure the `.vscode` directory is located in the parent directory of "Neural-Networks-From-Scratch-in-CPP" and "A Real Dataset".

2. **Configuration:**
   - Verify include, bin, and linking paths for libraries (e.g., NumPy, Python, OpenCV, Cereal) are correctly set in the `.vscode`, CMake files, and environment variables.

3. **Compile and Run:**
   - Use `Ctrl + Shift + B` in VSCode to build the project.
   - For applications using `matplotlibcpp`, update `run_my_app.bat` with the `.cpp` filename and execute `./run_my_app`.
   - For other applications, run `./filename`.

### A Real Dataset

#### Overview

This directory contains code for testing the classification neural network on the Fashion MNIST dataset:

- **Dataset**: Fashion MNIST
- **Description**: Consists of 60,000 training samples and 10,000 test samples, each a 28x28 grayscale image of a fashion item (e.g., shirt, dress, sneaker). It serves as a challenging replacement for the MNIST dataset. Remeber to unzip the dataset.

#### Setup and Compilation

1. **Unzip the Dataset:**
   - Ensure to unzip the `fashion_mnist_images.zip` file included in the repository. Extract the contents into the same directory where your project files are located.

2. **Install Required Tools and Libraries:**

   - **MSYS2 and MinGW-w64:**
     1. Install MSYS2.
     2. Use MSYS2 to install MinGW-w64 with POSIX threads.
     3. Add MSYS2 MinGW paths to environment variables.

   - **OpenCV:**
     1. Install MinGW build of OpenCV.
     2. Add OpenCV `bin` path to environment variables.

   - **Cereal:**
     1. Unzip the Cereal library.
     2. Add Cereal include path to the CMake file.

3. **Compile Using CMake:**

   1. Create and navigate to the `build` directory:
      ```bash
      mkdir build
      cd build
      ```

   2. Configure the project with CMake:
      ```bash
      cmake -DSOURCE_FILE_ARG="your_filename.cpp" -G "MinGW Makefiles" ..
      ```

   3. Delete `CMakeCache.txt` for a clean build:
        ```bash
        del CMakeCache.txt
        ```

   4. Build the project:
      ```bash
      cmake --build .
      ```

   5. Navigate to the executable directory and run:
      ```bash
      cd "A real dataset"
      ./run_my_plot (if using matplotlibcpp)
      ./MyProgram.exe (otherwise)
      ```

## Configuration Files

- **`.vscode/c_cpp_properties.json`**: IntelliSense configuration, including include paths and compiler settings.
- **`.vscode/settings.json`**: VSCode settings for file associations and CMake configuration.
- **`.vscode/tasks.json`**: Task configuration for building the project with `g++`.

- **`CMakeLists.txt`**: CMake configuration file for project setup, library finding, and compiler options.

Ensure all paths in these configuration files are updated to reflect your local setup. Delete `CMakeCache.txt` when compiling new files to avoid conflicts. Some of the required setup files and zip files are included in the repository.

## Acknowledgments

This project is based on concepts from **"Neural Networks from Scratch in Python"**. The book provided foundational understanding of neural networks, and this implementation applies those principles in C++.

---

Feel free to reach out if you have any questions or need further assistance!
