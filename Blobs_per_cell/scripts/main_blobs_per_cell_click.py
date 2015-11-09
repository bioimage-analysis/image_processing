import blobs_per_cell_click as bpcc
import open_image_bioformat as oib

path = "/Users/Espenel/Desktop/analysis_geri/Analysis_Ecad/GFP-K370, MetOH fixed/GFP_K370_MetOH"
image_blob, im = oib.batch_analysis_bioformat(path)

result={'Spot in injected cells':[],'Spot in NON injected cells':[]}

for img in image_blob:

    channel1, channel2 = bpcc.define_ch1_ch2(img, im, channel1 = "channel2", channel2 = "channel0")
    channel1_channel2, channel2_thresh = bpcc.mask_numpy_array(channel1, channel2, thresh_o = 90)
    image_conv = bpcc.image_processing(channel1)
    blob_in_image_after_binary, blob_only_in_channel2, blob_not_in_channel2 = bpcc.blobs(image_conv, channel1_channel2, remove_mb = True, val = 160, size = 100)
    count = bpcc.numbercell(channel1, channel2_thresh)
    numb_cells_inj, numb_cells_tot = count.getnumbcells()

    try:
        blob_in_channel2 = float(len(blob_only_in_channel2)) / float(numb_cells_inj)
    except ZeroDivisionError:
        pass
    try:
        blob_not_in_channel2 = float(len(blob_not_in_channel2)) / float(numb_cells_tot-numb_cells_inj)
    except ZeroDivisionError:
        pass

    result['Spot in injected cells'].append(blob_in_channel2)
    result['Spot in NON injected cells'].append(blob_not_in_channel2)

bpcc.plot_result(path, result, save = True)

oib.end_javabridge()
