For using the function 'create_coco_dataset_json' in dop_dataset.py file, two folders need to be prepared: Images/ and Shapefile/. 

Image patches for training were put in the Images/ folder, and the corresponding annotations (here, bounding boxes shapefiles) were saved in Shapefile/ folder.

The 'get_image_extent' function reads the extent of image patches using gdal: (col, row, xmin, ymax, y_r, x_r), here col, row are the image width and height, (xmin, ymax) is the upper left coordinates, and y_r and x_r are the pixel size.
The bounding boxes are read by geopands, and return the coordinates: (minX, minY, maxX, maxY). 

The function uses image extent and coverts the bounding boxes coordinates into the coco bounding box format, which is (pixel_x_upper_left, pixel_y_upper_left, width, height)
And read the pathes of image patches and build a connection with image patch and annotations (bounding boxes) using image id, below is the format of coco json file:

{

"images": 
[
        {
            "id": 1,
            "width": 256,
            "height": 256,
            "file_name": "Dataset/Images\\2013/dop20rgb_32_594_5724_2_ni_2013-07-08_107.tif"
        },
        {}, {},
],

"annotations": 
[
{
            "id": 1,
            "image_id": 2,
            "category_id": 1,
            "bbox": [
                161,
                54,
                28,
                32
            ],
            "area": 896,
            "iscrowd": 0
        }, {}, {},],

"categories": 
[
        {
            "id": 1,
            "name": "StandingDeadTree",
            "supercategory": "none"
        }
]

}
