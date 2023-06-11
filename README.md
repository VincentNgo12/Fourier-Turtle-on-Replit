# Fourier-Turtle-on-Replit
This repo is for my CS20 assignment, I have to make this repo because replit requires cofiurations in order to use OpenCV on Python.

We're on our Python Turtle assignment and I have also wanted to implement the Discrete Fourier Transform (DFT) to approximate closed curves.

This program uses OpenCV to open an image file (Fourier_sketch.jpg) and extract contours from it.
These contour coordinates will then be feed into the DFT computation function to get the Fourier coefficients.

The Fourier coefficients will then be sorted by magnitude, from largest to smallest.

Finally, the program uses Python Turtle to draw the approximation using the time-based function computed using the prior Fourier coefficients.
