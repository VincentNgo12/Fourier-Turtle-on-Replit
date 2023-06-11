import cv2 as cv
import numba as nb
from numpy import sin, cos, arctan2, sqrt
from numpy import pi as PI
from typing import List, Dict
from turtle import Turtle, Screen


# The main function
def main():
  # Get the list of contours
  contourCoordinates = loadImageAndExtractContours()
  # Comput DFT on each contour and append it to a list
  contours = []
  for contour in contourCoordinates:
    FourierCoefficients = ComputeDFT(contour)
    contours.append(FourierCoefficients)

  # Using the Fourier Coefficient from each contour, draw the approximation
  for contour in contours:
    FourierCoefficients = contour
    DrawParametricCurve(FourierCoefficients, turtleObject)



# Function to comput Discrete Fourier Transform (DFT), compiled using jit because Python is slow as heck!
@nb.jit(nopython=True)
def ComputeDFT(coordinates) -> List[Dict[str, float]]:
  # Number of coordinates, you should refer to the DFT formula.
  N = len(coordinates)
  FourierCoefficients = []

  # Calculating...
  for k in range(N):
    sum = complex(0,0)
    for n in range(N):
      angle = k * n * ((2*PI) / N)
      coordinateTerm = complex(coordinates[n][0],coordinates[n][1])
      FourierTerm = complex(cos(angle), -sin(angle))

      sum += coordinateTerm * FourierTerm

    coefficient = sum/N
    re = coefficient.real
    im = coefficient.imag
    maginitude = sqrt(re**2 + im**2)
    phase = arctan2(im,re)
    freq = k
    FourierCoefficients.append({"magnitude":maginitude, "phase":phase, "freq":freq})
    
    print(f"Calculation #{k}")

  FourierCoefficients.sort(key= lambda k: k["magnitude"], reverse=True)  
  return FourierCoefficients



# Function to draw with given Fourier Coefficients
def DrawParametricCurve(FourierCoefficients, turtleObject):
  N = len(FourierCoefficients)
  time = 0
  turtleObject.up()
  for t in range(N):
    x = 0
    y = 0
    for coefficient in FourierCoefficients:
      radius = coefficient["magnitude"]
      phase = coefficient["phase"]
      freq = coefficient["freq"]

      x += cos(freq * time + phase) * radius
      y += sin(freq * time + phase) * radius
      
    turtleObject.goto(x,y)
    turtleObject.down()
    time += (2 * PI) / N

  return


def loadImageAndExtractContours():
  # Load the image
  image = cv.imread("Fourier_sketch.jpg")
  # convert the image to grayscale
  image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  # Invert the image color
  image = cv.bitwise_not(image)
  
  # Apply a threshold to start finding contours
  ret, thresh = cv.threshold(image, 127, 255, 0)
  
  
  # Find contours in the image
  contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  
  # Create a list that contains each contour
  contourCoordinates = []
  for contour in contours:
    contour_points = contour[:, 0, :]  # Extract the points from the contour array
    
    # Convert contour points to a list of (x, y) tuples and append it to the contourCoordinates list
    contourCoordinates.append([(point[0]-300, -point[1]+300) for point in contour_points])

  # return the list
  return contourCoordinates


if __name__ == "__main__":
  # Ignore these please, just to supress some warnings raised by Numba, Python's speed is a pain in the bum.
  from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
  import warnings
  warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
  warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
  
  # Setting up the Turtle
  screen = Screen()
  screen.setup(800, 800)
  screen.bgcolor("white")
  turtleObject = Turtle()
  turtleObject.pensize(1)
  turtleObject.speed(0)
  turtleObject.up()
  # Run the big guy
  main()