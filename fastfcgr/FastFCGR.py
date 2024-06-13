import math
import numpy as np
import numexpr as ne
from PIL import Image
import timeit

class FastFCGR:
    def __init__(self):
        self.__sequence = ""
        self.__k = 0
        self.__matrix = None
        self.__maxValue = 0
        self.__currMatrixSize = 0

    #region getters
    @property
    def get_sequence(self):
        return self.__sequence

    @property
    def get_maxValue(self):
        return self.__maxValue
    
    @property
    def get_matrix_size(self):
        return self.__currMatrixSize
    
    @property
    def get_matrix(self):
        return self.__matrix
    #endregion

    #region readers
    def set_sequence_from_file(self, path:str, force: bool = False):       
        if not force and self.__sequence:
            raise Exception("Sequence already loaded. Use force=True to reload.")
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() and not line.startswith(('>', ';')):
                    self.__sequence += line.strip()

    def set_sequence(self, sequence:str, force: bool = False):       
        if not force and self.__sequence:
            raise Exception("Sequence already loaded. Use force=True to reload.")
        self.__sequence = sequence
    #endregion  

    def initialize(self, k):
        matrixSize = int(2 ** k)
        self.__currMatrixSize = matrixSize
        self.__matrix = np.zeros((matrixSize, matrixSize), dtype=np.uint)
        self.__maxValue = 0
        self.__k = k

    def calculate(self, scalingFactor:float=0.5):
        lastX, lastY = 0.0, 0.0
        self.__maxValue = 0

        for i in range(1, len(self.__sequence) + 1):
            base = self.__sequence[i - 1]
            if base not in 'aAcCgGtT': #TODO: check if this is the correct way to handle invalid bases
                continue

            dirX : float = 1 if base in 'tgTG' else -1
            dirY = 1 if base in 'atAT' else -1

            lastX += scalingFactor * (dirX - lastX)
            lastY += scalingFactor * (dirY - lastY)

            if(i < self.__k):
                continue

            # x = int(math.floor(np.nextafter((lastX + np.nextafter(1.0,0)),0) * self.__currMatrixSize / 2))
            # y = int(math.floor(np.nextafter(np.nextafter(1.0,0) - lastY,0) * self.__currMatrixSize / 2))
            x = int(math.floor((lastX + 1.0) * self.__currMatrixSize / 2))
            y = int(math.floor((1.0 - lastY) * self.__currMatrixSize / 2))
            self.__matrix[y, x] += 1

            if self.__matrix[y, x] > self.__maxValue:
                self.__maxValue = self.__matrix[y, x]
        return self.__maxValue
    
    def print_matrix(self):
        for row in self.__matrix:
            print(" ".join(f"{val:5}" for val in row))

    def save_image(self, path:str, d_max:int=255):
        normalized_matrix = FastFCGR.__rescale_interval(self.__matrix, self.__maxValue, d_max)
        grayscale_image = normalized_matrix.reshape((self.__currMatrixSize, self.__currMatrixSize))
        image = Image.fromarray(grayscale_image,mode=FastFCGR.__pillow_mode_from_bits(FastFCGR.__num_bits_needed(d_max)))        
        image.save(path)
    
    #region helpers
    @staticmethod
    def __num_bits_needed(n:int):
        return 1 if n == 0 else math.ceil(math.log2(n + 1))

    @staticmethod
    def __numpy_type_from_bits(bits:int):
        if bits <= 8:
            return np.uint8
        elif bits <= 16:
            return np.uint16
        else:
            raise ValueError("Number is too large to be represented by standard NumPy unsigned integer types.")
    
    @staticmethod
    def __pillow_mode_from_bits(bits:int):
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
        # mat = d_max - ((value / s_max) * d_max)
        mat = ne.evaluate("d_max - ((value / s_max) * d_max)")
        return mat.astype(FastFCGR.__numpy_type_from_bits(FastFCGR.__num_bits_needed(d_max)))
    #endregion 