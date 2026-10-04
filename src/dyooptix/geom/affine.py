import numpy as np


def build_affine(params: dict, centroid: tuple, width: int):
    """Build the 2x3 affine matrix for cv2.warpAffine.

    params   : {"tx", "ty", "cos", "sin", "log_s"}  (tx, ty as fractions of width)
    centroid : (cx, cy) in pixels
    width    : image width W in pixels
    returns  : float array of shape (2, 3)
    """
    # your code here

    tx = params["tx"]*width
    ty = params["ty"]*width

    cx,cy = centroid

    s = np.exp(params["log_s"])

    thetha = np.arctan2(params["sin"], params["cos"])

    a = s * np.cos(thetha)
    b = s * np.sin(thetha)

    arr = np.array([
            [a, b,  tx + (1-a)*cx - b*cy],
            [-b, a, ty + b*cx + (1-a)*cy]
        ], dtype = np.float64)

    return arr

    



    
