# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 23:09:08 2014
Thermal colormap in RGB space with rounded curve
https://gist.github.com/endolith/74275dc8fa2bb9a78266
"""
from __future__ import division
from numpy import pi, arange, zeros, ones, cos, sin, dstack, vstack
from matplotlib import cm


def smooth_hot(lutsize=1025):
    """
    Same as Matlab `hot`, but rounds the corners of the curve to avoid bright
    bands
    """
    r = 0.5
    some_len = 1-r

    # simplified:
    total_arc_length = pi*r - 4*r + 3

    dt = total_arc_length / (lutsize-1)

    b2r_t = arange(0, some_len, dt)
    b2r_rgbs = dstack((b2r_t, 0*b2r_t, 0*b2r_t))[0]

    remainder = some_len - b2r_t[-1]
    offset = dt - remainder

    dtheta = dt / r

    red_theta = arange(offset/r, pi/2, dtheta)
    red_rgbs = dstack(( r*sin(red_theta) + 1 - r,
                       -r*cos(red_theta) + r,
                       zeros(len(red_theta))))[0]

    remainder = r*(pi/2 - red_theta[-1])
    offset2 = dt - remainder

    r2y_t = arange(r+offset2, 1-r, dt)
    r2y_rgbs = dstack(( ones(len(r2y_t)),
                       r2y_t,
                       zeros(len(r2y_t))))[0]

    y_theta = arange(offset/r, pi/2, dtheta)
    y_rgbs = dstack(( ones(len(y_theta)),
                      r*cos(y_theta) + 1 - r,
                     -r*sin(y_theta) + r,
                      ))[0][::-1]

    y2w_t = arange(1, 1-some_len, -dt)[::-1]
    y2w_rgbs = dstack((ones(len(y2w_t)), ones(len(y2w_t)), y2w_t))[0]

    rgbs = vstack((b2r_rgbs, red_rgbs, r2y_rgbs, y_rgbs, y2w_rgbs))

    return cm.colors.LinearSegmentedColormap.from_list('smooth_hot', rgbs,
                                                       lutsize)