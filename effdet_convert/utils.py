

def decode_box(xmin, ymin, xmax, ymax, img_size, origin_w, origin_h):
    scale_w = origin_w/img_size
    scale_h = origin_h/img_size

    center_x = (xmax + xmin)/2
    center_y = (ymax + ymin)/2
    hw = (xmax - xmin)/2
    hh = (ymax - ymin)/2

    out_xmin = center_x*scale_w - hw*scale_w
    out_ymin = center_y*scale_h - hh*scale_h
    out_xmax = center_x*scale_w + hw*scale_w
    out_ymax = center_y*scale_h + hh*scale_h

    return [out_xmin, out_ymin, out_xmax, out_ymax]
