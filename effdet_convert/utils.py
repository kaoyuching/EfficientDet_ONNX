from typing import Tuple


def decode_box(xmin, ymin, xmax, ymax, img_size: Tuple, origin_w, origin_h):
    r"""
    img_size: (h, w)
    """
    scale_w = origin_w/img_size[1]
    scale_h = origin_h/img_size[0]

    center_x = (xmax + xmin)/2
    center_y = (ymax + ymin)/2
    hw = (xmax - xmin)/2
    hh = (ymax - ymin)/2

    out_xmin = center_x*scale_w - hw*scale_w
    out_ymin = center_y*scale_h - hh*scale_h
    out_xmax = center_x*scale_w + hw*scale_w
    out_ymax = center_y*scale_h + hh*scale_h

    return [out_xmin, out_ymin, out_xmax, out_ymax]
