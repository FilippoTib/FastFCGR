import math
import numpy as np
from PIL import Image

class FastFCGR:
    """
    Class for generating a graphical representation using the "Frequency Chaos Game Representation" (FCGR)
    from a nucleotide sequence (DNA or RNA).

    This class allows you to:
      - Load a nucleotide sequence from a file or directly from a string.
      - Initialize a matrix of size 2^k x 2^k.
      - Compute the FCGR matrix using a geometrical algorithm with a scaling factor.
      - Display the matrix and save the result as an image.
    """
    
    def __init__(self):
        """
        Initializes the class attributes.
        
        Attributes:
            __sequence (list): List that will contain the nucleotide sequence characters.
            __k (int): K-mer size used to calculate the matrix size (2^k).
            __matrix (np.ndarray): Matrix that will be populated with FCGR (k-mer counts).
            __isRNA (bool): Flag indicating if the sequence is RNA (True) or DNA (False).
            __maxValue (int): Maximum value found in the matrix (used for normalization).
            __currMatrixSize (int): Current size of the matrix. 
        """
        
        self.__sequence = []
        self.__k = 0
        self.__matrix = None
        self.__isRNA = False
        self.__maxValue = 0
        self.__currMatrixSize = 0

    #region getters
    @property
    def get_sequence(self):
        """
        Returns the current nucleotide sequence.

        Returns:
            list: List of characters representing the sequence.
        """
        
        return self.__sequence

    @property
    def get_maxValue(self):
        """
        Returns the maximum value found in the matrix.

        Returns:
            int: Maximum value in the matrix.
        """
        
        return self.__maxValue
    
    @property
    def get_matrix_size(self):
        """
        Returns the current size of the matrix (2^k).

        Returns:
            int: Size of the matrix.
        """
        
        return self.__currMatrixSize
    
    @property
    def get_matrix(self):
        """
        Returns the matrix containing the CGR representation.

        Returns:
            numpy.ndarray: Matrix of counts.
        """
        
        return self.__matrix
    #endregion

    #region readers
    def set_sequence_from_file(self, path:str, force: bool = False):       
        """
        Loads a nucleotide sequence from a text file (FASTA format).

        Lines starting with '>' or ';' are ignored.
        
        Args:
            path (str): Path to the FASTA file containing the sequence.
            force (bool, optional): If True, forces reloading even if a sequence is already loaded.
                                    Defaults to False.

        Returns:
            int: Length of the loaded sequence.

        Raises:
            Exception: If a sequence is already loaded and force is False.
        """
        
        if not force and self.__sequence:
            raise Exception("Sequence already loaded. Use force=True to reload.")
        with open(path, 'r') as file:
            lines = file.readlines()
            s = [s.strip().upper() for s in lines if not s.startswith(('>', ';'))]
            self.__sequence= list(''.join(s))
        return len(self.__sequence)

    def set_sequence(self, sequence:str, force: bool = False):    
        """
        Sets the nucleotide sequence directly from a string.

        Args:
            sequence (str): The nucleotide sequence.
            force (bool, optional): If True, forces updating even if a sequence is already loaded.
                                    Defaults to False.

        Returns:
            int: Length of the set sequence.

        Raises:
            Exception: If a sequence is already loaded and force is False.
        """   
        
        if not force and self.__sequence:
            raise Exception("Sequence already loaded. Use force=True to reload.")
        self.__sequence = list(sequence)
        return len(self.__sequence)
    #endregion  

    def initialize(self, k, isRNA:bool=False):
        """
        Initializes the matrix for the CGR representation and sets basic parameters.

        The matrix size will be 2^k x 2^k. The data type used in the matrix is chosen based on k:
        np.uint16 is used if k > 8, otherwise np.uint32 is used.
        
        Args:
            k (int): K-mer size used to determine the matrix size (2^k).
            isRNA (bool, optional): If True, treats the sequence as RNA (using 'U' instead of 'T').
                                    Defaults to False.
        """
        
        matrixSize = int(2 ** k)
        self.__currMatrixSize = matrixSize
        self.__matrix = np.zeros((matrixSize, matrixSize), dtype=np.uint16 if k > 8 else np.uint32)
        self.__maxValue = 0
        self.__k = k
        self.__isRNA = isRNA

    def calculate(self, scalingFactor:float = 0.5):
        """
        Computes the FCGR representation from the nucleotide sequence.

        The algorithm iterates over the sequence and for each valid base, it updates the coordinates (lastX, lastY)
        using a scaling factor. The counts are then accumulated in a temporary data structure. Finally, the main matrix is updated
        with the counts and the maximum value is recorded.

        Args:
            scalingFactor (float, optional): Scaling factor for updating the coordinates.
                                             Defaults to 0.5.

        Returns:
            int: The maximum value found in the matrix, used for normalization.
        """
        
        self.__maxValue = 0
        lastX, lastY = 0.0, 0.0
        
        halfMatrixSize = self.__currMatrixSize / 2
        valid_bases = {'A', 'C', 'G'} | ({'U'} if self.__isRNA else {'T'})
        
        temp_matrix = {}
        for i in range(1, len(self.__sequence) + 1):
            base = self.__sequence[i - 1]
            if base not in valid_bases:
                continue

            dirX,dirY = 1 if base in {'T','G','U'} else -1, 1 if base in {'A','T','U'} else -1

            lastX += scalingFactor * (dirX - lastX)
            lastY += scalingFactor * (dirY - lastY)

            if(i < self.__k):
                continue                

            x,y = min(math.floor((lastX + 1.0) * halfMatrixSize), self.__currMatrixSize - 1),min(math.floor((1.0 - lastY) * halfMatrixSize), self.__currMatrixSize - 1)
                
            if (y, x) in temp_matrix:
                temp_matrix[(y, x)] += 1
            else:   
                temp_matrix[(y, x)] = 1

            tmp_val = temp_matrix[(y, x)]
            if tmp_val > self.__maxValue:
                self.__maxValue = tmp_val
                    
        for (y, x), value in temp_matrix.items():
            self.__matrix[y, x] = value

        return self.__maxValue
    
    def print_matrix(self, path=None):
        """
        Prints the FCGR matrix or saves it to a file.

        If a file path is provided, the matrix is written to that file.
        Otherwise, it is printed to the console.

        Args:
            path (str, optional): Path to the file where the matrix will be saved.
                                  Defaults to None.
        """
        
        output = "\n".join(" ".join(f"{val:5}" for val in row) for row in self.__matrix)
        if path is not None:
            with open(path, "w") as file:
                file.write(output)
        else:
            print(output)

    def save_image(self, path:str, d_max:int=255):
        """
        Saves the FCGR as an image.

        The matrix is normalized to the range [0, d_max] and then converted to an image using Pillow.

        Args:
            path (str): Path where the image will be saved.
            d_max (int, optional): Maximum desired value for normalization.
                                   Defaults to 255.
        """
        
        normalized_matrix = FastFCGR.__rescale_interval(self.__matrix, self.__maxValue, d_max)
        image = Image.fromarray(normalized_matrix,mode=FastFCGR.__pillow_mode_from_bits(FastFCGR.__num_bits_needed(d_max)))        
        image.save(path)
    
    #region helpers
    @staticmethod
    def __num_bits_needed(n:int):
        """
        [PRIVATE] Calculates the number of bits required to represent the number n.

        Args:
            n (int): The number to be represented.

        Returns:
            int: Number of bits required.
        """
        
        return 1 if n == 0 else math.ceil(math.log2(n + 1))

    @staticmethod
    def __numpy_type_from_bits(bits:int):
        """
        [PRIVATE] Determines the appropriate NumPy data type based on the number of bits required.

        Args:
            bits (int): Number of bits required.

        Returns:
            dtype: NumPy data type (np.uint8 or np.uint16).

        Raises:
            ValueError: If the number of bits is greater than 16.
        """
        
        if bits <= 8:
            return np.uint8
        elif bits <= 16:
            return np.uint16
        else:
            raise ValueError("Number is too large to be represented by standard NumPy unsigned integer types.")
    
    @staticmethod
    def __pillow_mode_from_bits(bits:int):
        """
        [PRIVATE] Determines the Pillow image mode based on the number of bits.

        Args:
            bits (int): Number of bits.

        Returns:
            str: Pillow image mode ("1", "L", or "I;16").

        Raises:
            ValueError: If the number of bits is greater than 16.
        """
        
        if bits <= 1:
            return "1"     
        elif bits <= 8:
            return "L"     
        elif bits <= 16:
            return "I;16"  
        else:
            raise ValueError("Number is too large to be represented by standard Pillow image modes.")
        
    @staticmethod
    def __rescale_interval(value, s_max:int, d_max:int):            
        """
        Normalizes the values in the matrix to the interval [0, d_max].

        Args:
            value (numpy.ndarray): Original matrix.
            s_max (int): Maximum value in the original matrix.
            d_max (int): Maximum desired value after normalization.

        Returns:
            numpy.ndarray: Normalized matrix cast to the appropriate data type.
        """ 
        
        mat = d_max - ((value / s_max) * d_max)    
        return mat.astype(FastFCGR.__numpy_type_from_bits(FastFCGR.__num_bits_needed(d_max)))
    #endregion 