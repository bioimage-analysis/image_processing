#!/usr/bin/python -tt

import ImageAlignment as ai
import open_image_alignment as oia


def main():

    pathTObeads = "/.../.../beads.nd2"
    matrix = ai.define_matrix(pathTObeads)

    oia.overlay_channels(pathTObeads)

    pathTOanalyse = "/.../.../.../"
    images, imList = oia.batch_analysis_bioformat(pathTOanalyse)

    image_aligned = [ai.align(im, matrix) for im in imList]

    oia.overlay_channels(image_aligned[0])
    ai.imsave_al(image_aligned, imList[0], pathTOanalyse)
    oia.end_javabridge()

if __name__ == '__main__':
  main()
