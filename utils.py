import os
import json
import shutil
from glob import glob
import numpy as np
from skimage import measure
from PIL import Image, ImageDraw
from osgeo import gdal, ogr, gdal_array
from matplotlib import pyplot as plt
from matplotlib import patches
from pycocotools import mask as M

# dataset root dir
dir = r"../../../../35_plots/"
# img_path = dir + r"images/WV3_2016-11_north_ha_plots_1.tif "
# vec_path = dir + r"vectors/hectare_plots_treecrowns_1.shp"
# msk_path = dir + r"masks/hectare_plots_treecrowns_1_instance.tif"
# ano_path = dir + r"annotations/hectare_plots_treecrowns_1_instance.txt"


def vector_rasterization(image_path, vector_path):
    # get image raster info
    image_dataset = gdal.Open(image_path)
    col = image_dataset.RasterXSize
    row = image_dataset.RasterYSize
    image_geotrans = image_dataset.GetGeoTransform()
    ulY, ulX, distY, distX = image_geotrans[3], image_geotrans[0], \
                             abs(image_geotrans[5]), abs(image_geotrans[1])

    def world2Pixel(x, y):
        pixel_x = abs(int((x - ulX) / distX))
        pixel_y = abs(int((ulY - y) / distY))
        pixel_y = row if pixel_y > row else pixel_y
        pixel_x = col if pixel_x > col else pixel_x
        return pixel_x, pixel_y

    # read vector and then rasterize
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(vector_path, 0)
    layer = dataSource.GetLayer(0)

    rasterPoly = Image.new('I', (col, row), 0)
    rasterization = ImageDraw.Draw(rasterPoly)

    feature_num = layer.GetFeatureCount()

    for i in range(feature_num):
        points = []
        pixels = []
        feature = layer.GetFeature(i)
        geom = feature.GetGeometryRef()
        feature_type = geom.GetGeometryName()
        if feature_type == 'POLYGON' or 'MULTIPOLYGON':
            for j in range(geom.GetGeometryCount()):
                sub_polygon = geom.GetGeometryRef(j)
                if feature_type == 'MULTIPOLYGON':
                    sub_polygon = sub_polygon.GetGeometryRef(0)
                area = sub_polygon.GetArea()
                # print(area)
                if area > 4:
                    for p_i in range(sub_polygon.GetPointCount()):
                        px = sub_polygon.GetX(p_i)
                        py = sub_polygon.GetY(p_i)
                        points.append((px, py))

                    for p in points:
                        origin_pixel_x, origin_pixel_y = world2Pixel(
                            p[0], p[1])
                        # the pixel in new image
                        new_pixel_x, new_pixel_y = origin_pixel_x, origin_pixel_y
                        pixels.append((new_pixel_x, new_pixel_y))
                    rasterization.polygon(pixels, i+1)
                    pixels = []
                    points = []
        else:
            pass
    mask = np.array(rasterPoly)
    return mask, image_dataset


def crop_raster(image_path, crop_size, output_dir):
    file_name = image_path.split('/')[-1].split('.')[0]
    image_dataset = gdal.Open(image_path)
    col = image_dataset.RasterXSize
    row = image_dataset.RasterYSize
    row_num = int(row / crop_size)
    col_num = int(col / crop_size)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 1
    for i in range(row_num):
        for j in range(col_num):
            image_block = np.array(image_dataset.ReadAsArray(j*crop_size, i*crop_size, crop_size, crop_size))
            image_block = image_block.astype(np.float32)

            try:
                save_path = os.path.join(output_dir, f'{file_name}_{(i*col_num+j)}.tif')
                save_raster(image_block, image_dataset, save_path,
                            offset_x=j*crop_size, offset_y=i*crop_size)
                count += 1
            except Exception:
                raise IOError('crop failed %d' % count)


def save_raster(array, original_ds, save_path, offset_x=0, offset_y=0):
    ds = gdal_array.OpenArray(array)
    gdal_array.CopyDatasetInfo(original_ds, ds, xoff=offset_x, yoff=offset_y)

    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(save_path, ds)

    if len(array.shape) == 3:
        for i in range(array.shape[0]):
            ds.GetRasterBand(i + 1).WriteArray(array[i])
    else:
        ds.GetRasterBand(1).WriteArray(array)
    del ds
    print(f'save image into {save_path}')


def covert_mask_to_annotations(mask_path, tolerance=0, txt=None):
    def close_contour(cont):
        if not np.array_equal(cont[0], cont[-1]):
            cont = np.vstack((cont, cont[0]))
        return cont

    bbox, area, annotations = [], [], []
    # bbox = []
    # mask = np.asarray(Image.open(mask_path)).astype(np.int8)
    mask = get_image(mask_path)
    instances_list = np.unique(mask)
    for ins_id in instances_list[1:]:
        ins_mask = np.zeros(mask.shape).astype(np.int8)
        ins_mask[np.where(mask == ins_id)] = 255

        binary_mask_encoded = M.encode(
            np.asfortranarray(ins_mask.astype(np.uint8)))

        ins_mask = np.asarray(Image.fromarray(ins_mask).convert('1')).astype(np.uint8)

        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_ins_mask = np.pad(
            ins_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_ins_mask, 0.5)

        # contours = np.asarray(contours, dtype=object)

        contours = np.subtract(contours, 1)
        for contour in contours:
            contour = close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

        if len(polygons) > 0:
            annotations.append(polygons[0])
            bbox.append(M.toBbox(binary_mask_encoded))
            area.append(M.area(binary_mask_encoded))
    # if txt:
    #     filename = txt
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         for annot in annotations:
    #             f.write(f'1 ')
    #             for a in annot[0]:
    #                 f.write("%s " % (a / 320))
    #             f.write('\n')
    return bbox, area, annotations


def get_image(raster_path):
    ds = gdal.Open(raster_path)
    image = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b).ReadAsArray()
        image[:, :, b-1] = band
    if image.shape[-1] == 1:
        image = image[:, :, 0]
    else:
        image = norma_data(image, norma_methods='min-max')
    return image


def norma_data(data, norma_methods="dw"):
    arr = np.empty(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        array = data[:, :, i]
        mi_1, ma_99, mi_30, ma_70 = np.percentile(array, 1), np.percentile(array, 99), \
                                    np.percentile(array, 30), np.percentile(array, 70)
        if norma_methods == "dw":
            new_array = np.log(array * 0.0001 + 1)
            new_array = (new_array - mi_30 * 0.0001) / (ma_70 * 0.0001)
            new_array = np.exp(new_array * 5 - 1)
            new_array = new_array / (new_array + 1)

        else:
            new_array = (1*(array-mi_1)/(ma_99-mi_1)).clip(0, 1)
        arr[:, :, i] = new_array
    return arr


def plot_samples(image_path, bbox, annotation, save=False):
    img = get_image(image_path)
    # if isinstance(annotation, str):
    #     with open(annotation) as f:
    #         lines = f.readlines()
    #     annot = [l.rstrip().split(' ') for l in lines]
    #     annotations = [np.array([float(c) * 320 for c in a[1:]]) for a in annot]
    # else:
    #     annotations = annotation

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img[:, :, [5, 3, 1]])
    for bb, anno in zip(bbox, annotation):
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        polygon = np.array(anno).reshape(-1, 2)
        ax.add_patch(patches.Polygon(polygon,
                                     fill=True,
                                     alpha=.7,
                                     facecolor=c,
                                     lw=0))
        ax.add_patch(patches.Rectangle((bb[0], bb[1]), bb[2], bb[3],
                                       fill=False,
                                       color='r',
                                       lw=1))

    if save is False:
        plt.show()
    else:
        plt.savefig(dir + 'pngs/' + image_path.split('/')[-1].split('.')[0] + '.png')
    plt.close()


def convert_tif_jpeg(raster_path, save_path):
    image_array = get_image(raster_path)
    # print(image_array.shape)
    array = image_array[:, :, [5, 3, 1]]
    # print(array.shape)
    image = Image.fromarray(np.uint8(array*255))
    image.save(save_path)


def coco_format_from_masks(dir):
    image_files_full_path = glob(dir + 'images/*.tif')
    mask_files_full_path = glob(dir + 'masks/*.tif')

    print(f'Length of files: {len(image_files_full_path), len(mask_files_full_path)}')

    coco_format = {
        "info": [],
        "licenses": [],
        "categories": [{"id": 0, "name": "background"},
                       {"id": 1, "name": "tree"}],
        "images": [],
        "annotations": []
    }
    # images

    for i, img_name in enumerate(image_files_full_path):
        image_id = i + 1
        file_name = os.path.basename(img_name)
        image = {
            'id': image_id,
            'width': 320,
            'height': 320,
            'file_name': file_name,
        }
        coco_format['images'].append(image)

    # masks

    annotation_id = 1
    for j, msk_name in enumerate(mask_files_full_path):
        image_id = j + 1
        bboxs, areas, segmentations = covert_mask_to_annotations(msk_name, tolerance=1)
        for b, a, se in zip(bboxs, areas, segmentations):
            annotation = {
                'segmentation': [se],
                'area': int(a),
                'iscrowd': 0 if int(a) < 10000 else 1,
                'image_id': image_id,
                'bbox': b.tolist(),
                'category_id': 1,
                'id': annotation_id
            }
            annotation_id += 1
            coco_format['annotations'].append(annotation)

    with open(f'{dir}coco_instances_image.json', 'w') as output_json_file:
        json.dump(coco_format, output_json_file)


def copy_data(input_path, id, num, mark='train'):
    img_path = 'images'
    ann_path = 'masks'

    if num != 0:
        list = os.listdir(input_path + '/' + img_path)
        ann_list = os.listdir(input_path + '/' + ann_path)
        if not os.path.isdir(input_path + '/' + mark + '/' + img_path):
            os.makedirs(input_path + '/' + mark + '/' + img_path)
        if not os.path.isdir(input_path + '/' + mark + '/' + ann_path):
            os.makedirs(input_path + '/' + mark + '/' + ann_path)

        for i in range(num):
            shutil.copy(input_path + '/' + img_path + '/' + list[id[i]], input_path + '/' + mark + '/' + img_path
                        + '/' + list[id[i]])
            print('From src: ' + img_path + '/' + list[id[i]] + ' =>dst:' + '/' + mark + '/' + img_path
                  + '/' + list[id[i]])

            shutil.copy(input_path + '/' + ann_path + '/' + ann_list[id[i]],
                        input_path + '/' + mark + '/' + ann_path + '/' + ann_list[id[i]])

        f = open(input_path + '/' + mark + '/' + mark + '.txt', 'w')
        f.write(str(id))
        f.close()


def slice(input_path):
    img_path = 'images'
    ann_path = 'masks'

    list = os.listdir(input_path + '/' + img_path)
    # ann_list = os.listdir(input_path + '/' + ann_path)
    num_list = len(list)

    img_id = np.arange(num_list)
    np.random.shuffle(img_id)
    n_train, n_eval = 300, 15
    train_id, eval_id = img_id[:n_train], img_id[n_train:]

    copy_data(input_path, train_id, n_train, 'train')
    copy_data(input_path, eval_id, n_eval, 'val')


if __name__ == '__main__':
    pass
    # input_path = "E:/Repos/Bangalore/35_plots/315_patches"
    # slice(input_path)
    # coco_format_from_masks(dir="E:/Repos/Bangalore/35_plots/315_patches/train/")
    # coco_format_from_masks(dir="E:/Repos/Bangalore/35_plots/315_patches/val/")

    # n = np.random.randint(0, 330)
    # dir = r'E:/Repos/Bangalore/35_plots/315_patches/'
    # dir = r'E:/Repos/Bangalore/23_plots/size_320/'
    # # dir = r'E:/Repos/Bangalore/330_plot/size_320/'
    #
    # # image_dir = r'images/*.tif'
    # # masks_dir = r'masks/*.tif'
    # #
    # # for im in glob(dir + image_dir)[4:8]:
    # #     print(im)
    # #     img = get_image(im)
    # #     plt.imshow(img[:, :, [3, 1, 0]])
    # #     plt.show()
    # #     plt.imshow(img[:, :, [5, 3, 1]])
    # #     plt.show()
    # #     plt.hist(img[:, :, 2])
    # #     plt.show()
    #     # break
    #
    # # image_file_paths = glob(dir + image_dir)
    # # mask_file_paths = glob(dir + masks_dir)
    # #
    # # for im, ms in zip(image_file_paths, mask_file_paths):
    # #     print(im, ms)
    #     # crop_raster(image_path=im, crop_size=320, output_dir=dir + 'size_320/')
    #     # crop_raster(image_path=ms, crop_size=320, output_dir=dir + 'size_320/')
    #     # break
    #     # bbox, area, annotations = covert_mask_to_annotations(ms, tolerance=1)
    #     # print(len(bbox), len(annotations))
    # #     # print(len(area), area[0])
    # #     plot_samples(im, bbox, annotations)
    # #     # print(' ')
    # #     # print(bbox[0].tolist(), [annotations[0]])
    # #     break
    # #
    # # coco_format_from_masks(dir)
    # f = open(dir + 'coco_instances_image.json')
    # coco_dict = json.load(f)
    # print(coco_dict.keys())
    # for img_info in coco_dict['images']:
    #     # print(img_info['file_name'])
    #     new_name = img_info['file_name'][7:]
    #     # print(new_name)
    #     img_info['file_name'] = new_name
    # with open(f'{dir}coco_instances_image.json', 'w') as output:
    #     json.dump(coco_dict, output)
    # areas = [x['area'] for x in coco_dict['annotations']]
    # crowd = [x['iscrowd'] for x in coco_dict['annotations']]
    # #
    # print(len(areas), len(crowd))
    # small_crown = [x for x in areas if x <= 100]
    # medium_crown = [x for x in areas if 100 < x <= 900]
    #
    # print(f'small crown trees: {len(small_crown)}, around {len(small_crown)/len(areas)}')
    # print(f'medium crown trees: {len(medium_crown)}, around {len(medium_crown)/len(areas)}')

    # unique, counts = np.unique(crowd, return_counts=True)
    # print(unique, counts)

    #
    # counts, bins = np.histogram(crowd)
    # plt.stairs(counts, bins)
    # plt.show()
    # print(counts, bins)

    # clip vector
    # vector_paths = glob(dir + 'vectors/*.shp')
    # # image_paths = glob(dir + 'size_320/images/*.tif')
    # for v in vector_paths:
    #     # print(v)
    #     number = v.split('_')[-1].split('.')[0]
    #     # print(number)
    #     image_path = dir + f'size_320/images/tile_{number}_0.tif'
    #     # print(image_path)
    #
    #     SI = gdal.Open(image_path, gdal.GA_ReadOnly)
    #     GT = SI.GetGeoTransform()
    #
    #     left, top = gdal.ApplyGeoTransform(GT, 0, 0)
    #     right, bottom = gdal.ApplyGeoTransform(GT,
    #                                            SI.RasterXSize,
    #                                            SI.RasterYSize)
    #     # print(left, top, right, bottom)
    #     output = dir + f'size_320/vectors/mask_{number}_0.shp'
    #     os.system(f'ogr2ogr -clipdst {left} {bottom} {right} {top} {output} {v}')
        # break
