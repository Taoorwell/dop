import os
import json
import numpy as np
from osgeo import gdal
import geopandas as gpd


def download_images_and_extract_patches_shapefiles(y):

    grids_256p_year_path = f'Grids_256p_{y}.shp'
    grids_256p_year_bbox_path = f'Grids_256p_{y}_bboxes.shp'

    grids_256p_year = gpd.read_file(grids_256p_year_path)
    harz_dop_grids = gpd.read_file(r'Harz_dop_grids.shp')

    if not os.path.exists(f'Dataset/Images/{y}'):
        os.makedirs(f'Dataset/Images/{y}')
    if not os.path.exists(f'Dataset/Bboxes/{y}'):
        os.makedirs(f'Dataset/Bboxes/{y}')
    if not os.path.exists(f'/mnt/nas/dop/dop_{y}_rgbi'):
        os.makedirs(f'/mnt/nas/dop/dop_{y}_rgbi')

    for index, row in grids_256p_year.iterrows():
        for inde, ro in harz_dop_grids.iterrows():
            grid = row['geometry']
            dop_grids = ro['geometry']
            dop_image_name = ro['rgbi'].split('/')[-1]
            # rgb or rgbi, 2013 or 2022, 2016, 2019
            if grid.intersects(dop_grids) and dop_image_name.split('_')[-1].split('-')[0] == f'{y}':
                dop_image_path = f'/mnt/nas/dop/dop_{y}_rgbi/{dop_image_name}'
                dop_image_patch_path = f'Dataset/Images/{y}/' + dop_image_name.split('.')[0] + f'_{index}' + '.tif'
                if not os.path.exists(dop_image_path):
                    print(f'Download {dop_image_name}')
                    os.system(f"wget -P /mnt/nas/dop/dop_{y}_rgbi {ro['rgbi']}")

                (minX, minY, maxX, maxY) = grid.bounds
                gdal.Translate(dop_image_patch_path,
                               dop_image_path,
                               projWin=[minX, maxY, maxX, minY],
                               width=256, height=256,
                               resampleAlg=gdal.GRA_Cubic,
                               format='GTiff')

                dop_bbox_patch_path = f'Dataset/Bboxes/{y}/' + dop_image_name.split('.')[0] + f'_{index}' + '.shp'
                gdal.VectorTranslate(dop_bbox_patch_path,
                                     grids_256p_year_bbox_path,
                                     format='ESRI Shapefile',
                                     spatFilter=(minX, minY, maxX, maxY))


def get_image_extent(image_path):
    # Open the image using GDAL
    dataset = gdal.Open(image_path)
    # Get size of image
    col = dataset.RasterXSize
    row = dataset.RasterYSize
    # Get geotransform
    geotransform = dataset.GetGeoTransform()
    # Calculate the extent (coordinates of the corners)
    # (xmin, ymax) is the top-left corner
    xmin, ymax, y_r, x_r = geotransform[0], geotransform[3], abs(geotransform[5]), abs(geotransform[1])

    return col, row, xmin, ymax, y_r, x_r


def get_image_array(image_path):
    ds = gdal.Open(image_path)
    image = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b).ReadAsArray()
        image[:, :, b-1] = band
    return image


def create_coco_dataset_json(image_dir, shapefile_dir, output_json_path):
    images = []
    annotations = []
    categories = [{"id": 1, "name": "StandingDeadTree", "supercategory": "none"}]  # Single category example

    annotation_id = 1  # Unique ID for each annotation
    image_id = 1  # Unique ID for each image

    for year_folder in os.listdir(image_dir):
        for image_file in os.listdir(os.path.join(image_dir, year_folder)):
            image_path = os.path.join(image_dir, f'{year_folder}/{image_file}')
            shapefile_name = os.path.splitext(image_file)[0] + ".shp"
            shapefile_path = os.path.join(shapefile_dir, f'{year_folder}/{shapefile_name}')

            if not os.path.exists(shapefile_path):
                print(f"Shapefile for {image_file} not found. Skipping.")
                continue
            n_col, n_row, xmin, ymax, y_r, x_r = get_image_extent(image_path)
            # Add image metadata to COCO
            images.append({
                "id": image_id,
                "width": n_col,
                "height": n_row,
                "file_name": image_path
            })

            # Load shapefile and extract bounding boxes
            gdf = gpd.read_file(shapefile_path)
            for _, row in gdf.iterrows():
                (minX, minY, maxX, maxY) = row['geometry'].bounds
                pixel_x_upper_left = max(0, min(int((minX - xmin) / x_r), n_col))
                pixel_y_upper_left = max(0, min(int((ymax - maxY) / y_r), n_col))
                width = int((maxX - minX) / x_r)
                height = int((maxY - minY) / y_r)
                bbox = [pixel_x_upper_left, pixel_y_upper_left, width, height]

                # Add annotation metadata to COCO
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Assuming a single category
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                annotation_id += 1

            image_id += 1

    # Create final COCO dictionary
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save as JSON
    with open(output_json_path, "w") as json_file:
        json.dump(coco_dict, json_file, indent=4)

    print(f"COCO JSON saved to {output_json_path}")


if __name__ == '__main__':
    print('Download RBGI Images and Extracting Image Patches and Annotations')
    for y in [2013, 2016, 2019, 2022]:
        print(f'Year: {y}')
        download_images_and_extract_patches_shapefiles(y)
        print(f'Year: {y} Finished')
        print('')

    print('Creating COCO Json File for Dataset')
    create_coco_dataset_json(image_dir=r'Dataset/Images',
                             shapefile_dir=r'Dataset/Bboxes',
                             output_json_path='Dataset/coco_dataset.json')
