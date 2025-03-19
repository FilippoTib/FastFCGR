# FastFCGR

FastFCGR class for Frequency Chaos Game Representation

## Overview
The `FastFCGR` class generates a Frequency Chaos Game Representation (FCGR) for biological sequences (DNA or RNA).
It computes a matrix representing the distribution of nucleotide bases within the sequence using a geometrical algorithm based on fractal movements.

## Usage Guide

### Example
```python
# Import the class
from fast_fcgr import FastFCGR  # Ensure the file is named fast_fcgr.py

# Create an instance of the class
fcgr = FastFCGR()

# Load the sequence: you can load it directly or from a file.
# Example 1: Set the sequence directly
sequence = "ACGTACGTGACG"
fcgr.set_sequence(sequence)

# Example 2: Load the sequence from a file (text format)
# path = "path/to/sequence.txt"
# fcgr.set_sequence_from_file(path)

# Initialize parameters:
# k determines the matrix size (2^k x 2^k)
# isRNA indicates if the sequence is RNA (True) or DNA (False)
fcgr.initialize(k=5, isRNA=False)

# Calculate the FCGR matrix applying a scaling factor (default is 0.5)
max_value = fcgr.calculate(scalingFactor=0.5)
print("Maximum value in the matrix:", max_value)

# Print the calculated matrix
fcgr.print_matrix()

# Save the matrix as an image
fcgr.save_image("fcgr_output.png", d_max=255)

```

### Method Documentation

#### Properties (Getters)

##### `get_sequence`
- **Description**: Returns the loaded sequence as a list of characters.
- **Parameters**: None.

##### `get_maxValue`
- **Description**: Returns the maximum value present in the calculated FCGR matrix.
- **Parameters**: None.

##### `get_matrix_size`
- **Description**: Returns the current matrix size (i.e., the number of rows/columns, computed as 2^k).
- **Parameters**: None.

##### `get_matrix`
- **Description**: Returns the FCGR matrix as a NumPy array.
- **Parameters**: None.


#### Public Methods

##### `set_sequence_from_file(path: str, force: bool = False)`
- **Description**: Loads a sequence from a text file. Lines starting with `>` or `;` are ignored.
- **Parameters**:
  - `path`: A string specifying the file path.
  - `force`: Optional boolean flag; if `False` and a sequence is already loaded, an exception is raised. Set to `True` to force reloading.
- **Returns**: The length of the loaded sequence.

##### `set_sequence(sequence: str, force: bool = False)`
- **Description**: Directly sets the sequence from a given string.
- **Parameters**:
  - `sequence`: A string containing the DNA/RNA sequence.
  - `force`: Optional boolean flag; if `False` and a sequence is already loaded, an exception is raised. Set to `True` to force reloading.
- **Returns**: The length of the loaded sequence.

##### `initialize(k: int, isRNA: bool = False)`
- **Description**: Initializes the internal parameters and creates a matrix of size `2^k x 2^k`.
- **Parameters**:
  - `k`: An integer that defines the granularity of the matrix (matrix size = 2^k).
  - `isRNA`: Boolean flag indicating whether the sequence is RNA (if `True`, valid nucleotides include 'U' instead of 'T').

##### `calculate(scalingFactor: float = 0.5)`
- **Description**: Computes the FCGR matrix based on the loaded sequence and the provided scaling factor. The maximum value in the matrix is updated during the calculation. The minimum value in the matrix, on the other hand, is ALWAYS taken to be equal to 0.
- **Parameters**:
  - `scalingFactor`: A float value that determines the scaling factor for updating the coordinates (default is 0.5).
- **Returns**: The maximum value present in the FCGR matrix.

##### `print_matrix()`
- **Description**: Prints the FCGR matrix in a formatted text layout.
- **Parameters**: None.

##### `print_matrix(path: string = None)`
- **Description**: Prints the FCGR matrix in a formatted text layout. If the optional `path` parameter is provided, the output will be written to the specified file instead of being printed to the screen.
- **Parameters**:
  - `path` (optional): A string representing the file path where the matrix should be saved. If `None`, the matrix is printed to stdout.

##### `save_image(path: str, d_max: int = 255)`
- **Description**: Saves the FCGR matrix as an image. Before saving, the matrix is normalized based on the `d_max` value.
- **Parameters**:
  - `path`: A string specifying the path (and filename) where the image will be saved.
  - `d_max`: The maximum scale value (default 255) used for normalizing the image.
