#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:19:23 2018

@author: opensource
"""

from __future__ import print_function
from PIL import Image, ImageChops, ImageEnhance
import sys, os
import threading
import argparse

parser = argparse.ArgumentParser(description="""
Performs Error Level Analysis over a directory of images
""")
parser.add_argument('--dir', dest='directory', required=True,
                    help='path to the directory containing the images')
parser.add_argument('--quality', dest='quality',
                    help='quality used by the jpeg crompression alg.',
                    default=90)

TMP_EXT = ".tmp_ela.jpg"
ELA_EXT = ".ela.png"
SAVE_REL_DIR = "generated"
threads = []
quality = 90

def ela(fname, orig_dir, save_dir):
    """
    Generates an ELA image on save_dir.
    Params:
        fname:      filename w/out path
        orig_dir:   origin path
        save_dir:   save path
    """
    basename, ext = os.path.splitext(fname)

    org_fname = os.path.join(orig_dir, fname)
    tmp_fname = os.path.join(save_dir, basename + TMP_EXT)
    ela_fname = os.path.join(save_dir, basename + ELA_EXT)

    im = Image.open(org_fname)
    im.save(tmp_fname, 'JPEG', quality=quality)

    tmp_fname_im = Image.open(tmp_fname)
    ela_im = ImageChops.difference(im, tmp_fname_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    ela_im.save(ela_fname)
    os.remove(tmp_fname)


def main ():
    args = parser.parse_args()
    dirc = args.directory
    quality = args.quality

    ela_dirc = os.path.join(dirc, SAVE_REL_DIR)

    print("Performing ELA on images at %s" % dirc)

    if not os.path.exists(ela_dirc):
        os.makedirs(ela_dirc)

    for d in os.listdir(dirc):
        if d.endswith(".jpg") or d.endswith(".jpeg"):
            thread = threading.Thread(target=ela, args=[d, dirc, ela_dirc])
            threads.append(thread)
            thread.start()

    for t in threads:
        t.join()

    print("Finished!")
    print("Head to %s/%s to check the results!" % (dirc, SAVE_REL_DIR))


if __name__ == '__main__':
    main()
else:
    print("This should'nt be imported.", file=sys.stderr)
sys.exit(1)