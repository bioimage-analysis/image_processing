#!/usr/bin/python -tt

from joblib import Parallel, delayed

def main():

    path = ".../experiment1"
    path2 = ".../experiment2"
    path3 = ".../experiment3"

    path_list = [path, path2, path3]


    def analysis(path):

        images = bpc.batch_analysis(path)
        number_of_cells = Parallel(n_jobs=4)(delayed(bpc.number_nucleus)(img[0]) for img in images)
        image_conv =  Parallel(n_jobs=4)(delayed(bpc.image_processing)(img[1]) for img in images)
        blob_in_image = Parallel(n_jobs=4)(delayed(bpc.blobs)(im, remove_mb = True) for im in image_conv)

        number_blobs = [len(blob) for blob in blob_in_image]

        blobs_per_cell = [a / float(b) for a,b in zip (number_blobs, number_of_cells)]

        return blobs_per_cell

    values = [analysis(path) for path in path_list]

    bpc.plot_result(path_list, values, name_exp ='result for', title = "your experiment", save = False)


if __name__ == '__main__':
  main()
