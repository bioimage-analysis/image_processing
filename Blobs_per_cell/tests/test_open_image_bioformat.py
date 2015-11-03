from Find_spot_Ecad_Fotine.scripts
import open_image_bioformat as oib
import numpy as np

image = "/Users/Espenel/Desktop/analysis_geri/mCh-K370, GFP-actin, DECMA-Cy5, actin-AMCA-002.nd2"

path = "/Users/Espenel/Desktop/analysis_geri/"



def test_begin_javabridge():
    oib.begin_javabridge()
    assert oib.JVM_BEGIN == True

def test_image_info():
    img = oib.image_info(image)
    assert ("xsize" in img)


def test_read_bioformat():
    img = oib.read_bioformat(image, resize = True)
    assert np.max(img.shape) == 1024

def test_batch_analysis_bioformat():
    images = oib.batch_analysis_bioformat(path)
    assert len(images) > 1

def test_show_series_all():

    images, im = oib.batch_analysis_bioformat(path)

    axes, number_plots =oib.show_series_all(images, path, im)
    assert axes == number_plots

def test_show_series_all_histo():

    images, im = oib.batch_analysis_bioformat(path)

    axes, number_plots = oib.show_series_all_histo(images, im)
    assert axes / 2 == number_plots

def test_show_chunk_series_all_histo():

    images, im = oib.batch_analysis_bioformat(path)

    chunks = oib.show_chunk_series_all_histo(images, path, im, size_chunk = 2)
    assert len(chunks) == (len(images) / 2)


def test_end_javabridge():
    oib.end_javabridge()
    assert oib.JVM_END == True
