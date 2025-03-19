import math
from enum import Enum
import numpy as np
import numexpr as ne
from PIL import Image

class FastFCGR:
    def __init__(self):
        self.__sequence = []
        self.__k = 0
        self.__matrix = None
        self.__isRNA = False
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
            s = [s.strip().upper() for s in lines if not s.startswith(('>', ';'))]
            self.__sequence= list(''.join(s))
        return len(self.__sequence)

    def set_sequence(self, sequence:str, force: bool = False):       
        if not force and self.__sequence:
            raise Exception("Sequence already loaded. Use force=True to reload.")
        self.__sequence = list(sequence)
    #endregion  

    def initialize(self, k, isRNA:bool=False):
        matrixSize = int(2 ** k)
        self.__currMatrixSize = matrixSize
        self.__matrix = np.zeros((matrixSize, matrixSize), dtype=np.uint16 if k > 8 else np.uint32)
        self.__maxValue = 0
        self.__k = k
        self.__isRNA = isRNA

    def calculate(self, scalingFactor:float = 0.5):
        self.__maxValue = 0
        lastX, lastY = 0.0, 0.0
        
        halfMatrixSize = self.__currMatrixSize / 2
        valid_bases = {'A', 'C', 'G'} | ({'U'} if self.__isRNA else {'T'})
        
        temp_matrix = {}
        for i in range(1, len(self.__sequence) + 1):
            base = self.__sequence[i - 1]
            if base not in valid_bases:
                continue

            # dirX = 1 if base in {'t','g','u','T','G','U'} else -1
            # dirY = 1 if base in {'a','t','u','A','T','U'} else -1
            dirX,dirY = 1 if base in {'T','G','U'} else -1, 1 if base in {'A','T','U'} else -1

            lastX += scalingFactor * (dirX - lastX)
            lastY += scalingFactor * (dirY - lastY)

            if(i < self.__k):
                continue                

            # x = int(math.floor((lastX + 1.0) * self.__currMatrixSize / 2))
            # y = int(math.floor((1.0 - lastY) * self.__currMatrixSize / 2))

            # x = x if x < self.__currMatrixSize else x-1
            # y = y if y < self.__currMatrixSize else y-1

            # x= min(math.floor((lastX + 1.0) * halfMatrixSize), self.__currMatrixSize - 1)
            # y = min(math.floor((1.0 - lastY) * halfMatrixSize), self.__currMatrixSize - 1)
            x,y = min(math.floor((lastX + 1.0) * halfMatrixSize), self.__currMatrixSize - 1),min(math.floor((1.0 - lastY) * halfMatrixSize), self.__currMatrixSize - 1)

            # self.__matrix[y, x] += 1
            # temp_matrix[(y, x)] = temp_matrix.get((y, x), 0) + 1

            # if self.__matrix[y, x] > self.__maxValue:
            #     self.__maxValue = self.__matrix[y, x]
                
            if (y, x) in temp_matrix:
                temp_matrix[(y, x)] += 1
            else:   
                temp_matrix[(y, x)] = 1

        # Aggiorna il massimo locale
            tmp_val = temp_matrix[(y, x)]
            if tmp_val > self.__maxValue:
                self.__maxValue = tmp_val
                    
        for (y, x), value in temp_matrix.items():
            self.__matrix[y, x] = value

        return self.__maxValue
    
    def print_matrix(self):
        for row in self.__matrix:
            print(" ".join(f"{val:5}" for val in row))

    def save_image(self, path:str, d_max:int=255):
        normalized_matrix = FastFCGR.__rescale_interval(self.__matrix, self.__maxValue, d_max)
        image = Image.fromarray(normalized_matrix,mode=FastFCGR.__pillow_mode_from_bits(FastFCGR.__num_bits_needed(d_max)))        
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
        mat = d_max - ((value / s_max) * d_max)
        # print(f"[{0, s_max}] -> [0, {d_max}] --- from ({np.min(value)}, {np.max(value)}, {np.mean(value)}) to ({np.min(mat)}, {np.max(mat)}, {np.mean(mat)})")       
        return mat.astype(FastFCGR.__numpy_type_from_bits(FastFCGR.__num_bits_needed(d_max)))
    #endregion 