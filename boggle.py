#!/usr/local/bin/python3

import sys
import cv2
import numpy as np
import imutils

# Orders the corners of a quad to go in the right order to apply a perspective transform
# (used in find_board)
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Finds a boggle board in an image. Returns an image of just the board
#
# WARNING: this finds the board by assuming that the board is the biggest yellow
# thing in the image. If your boggle board is not yellow, or there are yellow
# things in the background, this will fail
def find_board(img, verbose=False):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the min and max colors (in HSV) counted as yellow
    # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
    color_min = (15, 50, 0)
    color_max = (36, 255, 210)

    # Extract only yellow parts of input image
    yellow_mask = cv2.inRange(hsv_img, color_min, color_max)
    yellow_img = cv2.bitwise_and(img, img, mask=yellow_mask)

    # Convert image to grayscale to do image processing
    gray = cv2.cvtColor(yellow_img, cv2.COLOR_BGR2GRAY)

    # =============================================================
    #                    Clean Up Image
    # =============================================================

    # "Close" image - removes foreground noise
    # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel_close)

    # "Open" image - removes background noise
    # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img_open = cv2.morphologyEx(kernel_close,cv2.MORPH_OPEN,kernel_open)

    # =============================================================
    #                  Locate the Board
    # =============================================================

    # Threshold image - turns image from grayscale into binary
    thresh = cv2.adaptiveThreshold(close,255,0,1,19,4)

    # Detect contours in image
    contours,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Find biggest contour. This should be the outline of the board
    outline = max(contours, key=cv2.contourArea)

    # =============================================================
    #            Map the Board to a Square
    # =============================================================
    # We've found the board, but it's in a random place in the image
    # and is probably distorted by projection. We'll find the board's
    # corners and use those to map the board to a square image

    # Simplify contour to a quadrilateral. This should identify the corners of the board
    # https://stackoverflow.com/questions/41138000/fit-quadrilateral-tetragon-to-a-blob
    tol = 0.1
    epsilon = tol*cv2.arcLength(outline,True)
    quad = cv2.approxPolyDP(outline,epsilon,True)

    # If the curve doesn't get simplified to a quad, try again a few times.
    # The epsilon parameter determines how far from the original curve the
    # simplified curve is allowed to get. If we have too few vertices,
    # we need to get stricter. If we have too many vertices, we need to get more permissive
    count = 0
    while len(quad) != 4 and count < 10:
        print(len(quad), tol)
        if len(quad) < 4:
            tol *= 0.9
        else:
            tol *= 1.1
        epsilon = tol*cv2.arcLength(outline,True)
        quad = cv2.approxPolyDP(outline,epsilon,True)
        count += 1

    if len(quad) != 4:
        cv2.imshow("yellow_img", gray)
        cv2.waitKey(0)
        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)
        cv2.drawContours(img, [quad],-1,(0, 255, 0),3)
        cv2.imshow("img", img)
        cv2.waitKey(0)

        raise RuntimeError("find_board error: could not find bounding quad. Instead, found a " + str(len(quad)) + "-gon")

    # Put points in order to apply perspective projection
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    rect = order_points(quad[:,0,:])
    maxWidth = 500
    maxHeight = 500
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # Apply perspective projection to board
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    if (verbose):
        cv2.imshow("warped", warped)
        cv2.waitKey(0)

    return warped

# Takes in an overhead image of just the board, and returns 2d list of images of the individual letters
def find_letters(board, verbose = False):
    width = np.size(board, 1)

    # Convert image to grayscale to do image processing
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold image - turns image from grayscale into binary
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
    ret, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

    # "Open" image - removes background noise
    # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    img_open = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel_open)

    # "Close" image - removes foreground noise
    # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    close = cv2.morphologyEx(img_open,cv2.MORPH_CLOSE,kernel_close)

    if verbose:
        cv2.imshow("input", close)
        cv2.waitKey(0)

    # Detect all contours in image
    contours,hier = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # =============================================================
    #            Identify Contours of Letter Tiles
    # =============================================================

    # Take the contours whose enclosed area is closest to 1/16 of the total area
    target_area = int(1/16. * width * width)

    # TODO: is 0.75 a good number here?
    allowed_err = 0.75 * target_area

    tiles = []
    good_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if abs(area - target_area) < allowed_err:
            # print(cv2.contourArea(c), '\t', target_area, '\t', abs(area - target_area) / target_area)
            good_contours.append(c)
            # x, y, w, h = cv2.boundingRect(c)
            tiles.append(cv2.boundingRect(c))

    if len(tiles) != 16:
        cv2.drawContours(board, [contours],-1,(0, 255, 0),3)
        cv2.imshow("board", board)
        cv2.imshow()
        raise RuntimeError("find_letters error: could not find 16 letters. I found " + str(len(tiles)) + " letters")

    # =============================================================
    #            Sort Tiles by Position
    # =============================================================

    # sorts tiles lexicographically. For our purposes, this just sorts the tiles by x component
    sorted_tiles = sorted(tiles)

    tile_images = []
    for i in range(4):
        tile_row = sorted_tiles[4 * i: 4 * i + 4]
        # sort row by y coordinate
        sorted_row = sorted(tile_row, key = lambda x : x[1])
        tile_images.append([])
        for (x, y, w, h) in sorted_row:
            crop_img = board[y:y+h, x:x+w]
            tile_images[i].append(crop_img)

    if verbose:
        cv2.drawContours(board, good_contours, -1, (0,255,0), 3)
        cv2.imshow("contours", board)
        cv2.waitKey(0)

    return tile_images

# Tries to match an image against a template symbol at a variety of scales.
# This is useful because we don't know exactly how bit the letter will be in our image.
# https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
# Returns how well the template matched the image at the best scale
def multiscale_match(img, template):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    template = imutils.resize(template, width= np.size(img, 0))
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_fit = 0
    for scale in np.linspace(0.1, 2.0, 10)[::-1]:

        scaled_img = imutils.resize(gray_img, width=int(gray_img.shape[1] * scale))
        r = gray_img.shape[1] / float(scaled_img.shape[1])
        if scaled_img.shape[0] < gray_template.shape[0] or scaled_img.shape[1] < gray_template.shape[1]:
            break

        res = cv2.matchTemplate(scaled_img, gray_template, cv2.TM_CCOEFF_NORMED)
        best_fit = max(best_fit, np.amax(res))

    return best_fit

# Tries to match an image against a template. Tries rotating the template by 90 degree
# rotations and rescaling it to find the best fit possible. Returns how good the best fit was
def match(img, template):
    angles = [0, 90, 180, 270]
    best_fit = 0
    for angle in angles:
        rotated_template = imutils.rotate_bound(template, angle)
        rotated_match = multiscale_match(img, rotated_template)
        best_fit = max(best_fit, rotated_match)

    return best_fit

def read_board(filename, verbose=False):
    img = cv2.imread(filename)
    img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    board = find_board(img)
    if verbose:
        cv2.imshow("board", board)
        cv2.waitKey(0)

    letter_tiles = find_letters(board)
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'qu', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    board_letters = []
    for tile_row in letter_tiles:
        letter_row = []
        for tile in tile_row:
            best_letter = 'a'
            best_letter_fit = 0
            for letter in letters:
                letter_img = cv2.imread('letters/' + letter + '.jpg')
                letter_fit = match(tile, letter_img)
                if letter_fit > best_letter_fit:
                    best_letter = letter
                    best_letter_fit = letter_fit
            letter_row.append(best_letter)
            if (verbose):
                print(best_letter, end=' ')
        board_letters.append(letter_row)
        if (verbose):
            print('')

read_board(sys.argv[1], True)
