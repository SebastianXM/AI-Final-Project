import zipfile
import os

'''
Data processing from UC Berkeley http://ai.berkeley.edu
'''

class Datum:
    """
    A datum is a pixel-level encoding of digits or face/non-face edge maps.

    Digits are from the MNIST dataset and face images are from the
    easy-faces and background categories of the Caltech 101 dataset.


    Each digit is 28x28 pixels, and each face/non-face image is 60x74
    pixels, each pixel can take the following values:
      0: no edge (blank)
      1: gray pixel (+) [used for digits only]
      2: edge [for face] or black pixel [for digit] (#)

    Pixel data is stored in the 2-dimensional array pixels, which
    maps to pixels on a plane according to standard euclidean axes
    with the first dimension denoting the horizontal and the second
    the vertical coordinate:

      28 # # # #      #  #
      27 # # # #      #  #
       .
       .
       .
       3 # # + #      #  #
       2 # # # #      #  #
       1 # # # #      #  #
       0 # # # #      #  #
         0 1 2 3 ... 27 28

    For example, the + in the above diagram is stored in pixels[2][3], or
    more generally pixels[column][row].

    The contents of the representation can be accessed directly
    via the getPixel and getPixels methods.
    """
    def __init__(self, data, width, height):
        """
        Create a new datum from file input (standard MNIST encoding).
        """
        DATUM_HEIGHT = height
        DATUM_WIDTH = width
        self.height = DATUM_HEIGHT
        self.width = DATUM_WIDTH
        if data == None:
            data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)]
        self.pixels = arrayInvert(convertToInteger(data))

    def getPixel(self, column, row):
        """
        Returns the value of the pixel at column, row as 0, or 1.
        """
        return self.pixels[column][row]

    def getPixels(self):
        """
        Returns all pixels as a list of lists.
        """
        return self.pixels

    def getAsciiString(self):
        """
        Renders the data item as an ascii image.
        """
        rows = []
        data = arrayInvert(self.pixels)
        for row in data:
            ascii = map(asciiGrayscaleConversionFunction, row)
            rows.append( "".join(ascii) )
        return "\n".join(rows)

    def __str__(self):
        return self.getAsciiString()
    
def asciiGrayscaleConversionFunction(value):
    """
    Helper function for display purposes.
    """
    if(value == 0):
        return ' '
    elif(value == 1):
        return '+'
    elif(value == 2):
        return '#'
    
def IntegerConversionFunction(character):
    """
    Helper function for file reading.
    """
    if(character == ' '):
        return 0
    elif(character == '+'):
        return 1
    elif(character == '#'):
        return 2

def convertToInteger(data):
    """
    Helper function for file reading.
    """
    if not isinstance(data, list):
        return IntegerConversionFunction(data)
    else:
        return list(map(convertToInteger, data))
    
def arrayInvert(array):
    """
    Inverts a matrix stored as a list of lists.
    """
    result = [[] for i in array]
    for outer in array:
        for inner in range(len(outer)):
            result[inner].append(outer[inner])
    return result

# Data processing, cleanup and display functions

def loadDataFile(filename, n,width,height):
    """
    Reads n data images from a file and returns a list of Datum objects.

    (Return less then n items if the end of file is encountered).
    """
    DATUM_WIDTH=width
    DATUM_HEIGHT=height
    fin = readlines(filename)
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < DATUM_WIDTH-1:
            # we encountered end of file...
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Datum(data,DATUM_WIDTH,DATUM_HEIGHT))
    return items

def readlines(filename):
    "Opens a file or reads it from the zip archive data.zip"
    if(os.path.exists(filename)):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).split('\n')

def loadLabelsFile(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels

def get_data():
    # load digits data
    digits_train_data = loadDataFile("data/digitdata/trainingimages", 5000, 28, 28)
    digits_train_labels = loadLabelsFile("data/digitdata/traininglabels", 5000)
    digits_test_data = loadDataFile("data/digitdata/testimages", 1000, 28, 28)
    digits_test_labels = loadLabelsFile("data/digitdata/testlabels", 1000)
    digits_val_data = loadDataFile("data/digitdata/validationimages", 1000, 28, 28)
    digits_val_labels = loadLabelsFile("data/digitdata/validationlabels", 1000)

    # load face data
    face_train_data = loadDataFile("data/facedata/facedatatrain", 451, 60, 70)
    face_train_labels = loadLabelsFile("data/facedata/facedatatrainlabels", 451)
    face_test_data = loadDataFile("data/facedata/facedatatest", 150, 60, 70)
    face_test_labels = loadLabelsFile("data/facedata/facedatatestlabels", 150)
    face_val_data = loadDataFile("data/facedata/facedatavalidation", 301, 60, 70)
    face_val_labels = loadLabelsFile("data/facedata/facedatavalidationlabels", 301)

    # create features
    digits_X_train = [[[0 if pixel == 0 else (1 if pixel == 1 else 2) for pixel in row] for row in data.getPixels()] for data in digits_train_data]
    digits_X_test = [[[0 if pixel == 0 else (1 if pixel == 1 else 2) for pixel in row] for row in data.getPixels()] for data in digits_test_data]
    digits_X_val = [[[0 if pixel == 0 else (1 if pixel == 1 else 2) for pixel in row] for row in data.getPixels()] for data in digits_val_data]

    face_X_train = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in face_train_data]
    face_X_test = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in face_test_data]
    face_X_val = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in face_val_data]
    face_X_train = [sample[:60] for sample in face_X_train]
    face_X_test = [sample[:60] for sample in face_X_test]
    face_X_val = [sample[:60] for sample in face_X_val]

    # create labelsp
    digits_y_train = digits_train_labels
    digits_y_test = digits_test_labels
    digits_y_val = digits_val_labels

    face_y_train = face_train_labels
    face_y_test = face_test_labels
    face_y_val = face_val_labels

    return digits_X_train, digits_y_train, digits_X_test, digits_y_test, digits_X_val, digits_y_val, face_X_train, face_y_train, face_X_test, face_y_test, face_X_val, face_y_val