from dataclasses import dataclass

import numpy, numpy as np
import SimpleITK, SimpleITK as sitk


@dataclass
class ImageTile:
    """ImageTile class that contains an image tile and indices that were used to extract the
    tile and that are necessary to locate the image tile in its original source image.
    
    Attributes
    ----------
    
    image: numpy.ndarray | SimpleITK.Image
        Image tile
    
    x_props: tuple[int, int, int, int, int, int]
        Tiling information on x-axis.
    
    y_props: tuple[int, int, int, int, int, int]
        Tiling information on y-axis.
    
    original_shape: tuple[int, int]
        Original image resolution
    """
    
    image: numpy.ndarray | SimpleITK.Image
    
    x_props: tuple[int, int, int, int, int, int]
    y_props: tuple[int, int, int, int, int, int]
    original_shape: tuple[int, int] | None = None
    

def get_tile_params(size: int, n_tiles: int = 2, overlap: float = 0.1) \
    -> tuple[numpy.ndarray, \
             numpy.ndarray, \
             numpy.ndarray, \
             numpy.ndarray, \
             numpy.ndarray, \
             numpy.ndarray]:
    """Gets indices for tiling in one dimension. Indices include start index in image,
    start index in image with overlap, start index in tile (considering overlap), end
    index in image, end index in image with overlap, end index in tile.
    

    Args:
        size (int): size of dimension
        n_tiles (int, optional): Number of tiles to return. Defaults to 2.
        overlap (float, optional): Overlap between tiles in percent. Defaults to 0.1.

    Returns:
        tuple[int, int, int, int, int, int]: Start and end indices for tiling.
    """
    tile_size = size // n_tiles
    overlap_offset = int(tile_size * overlap)

    starts = []
    starts_ov = []
    ends = []
    ends_ov = []
    starts_rel = []
    ends_rel = []

    for i in range(n_tiles):
        start = tile_size * i
        end = start + tile_size
        if i == n_tiles - 1:
            # Get dangling end pixels
            end += size % n_tiles
        starts.append(start)
        ends.append(end)
        
        start_ov = start - overlap_offset
        start_ov_oob = 0 - start_ov
        if start_ov_oob > 0:
            start_ov = start_ov + start_ov_oob
            start_rel = overlap_offset - start_ov_oob
        else:
            start_rel = overlap_offset
        
        end_ov = end + overlap_offset
        end_ov_oob = end_ov - size
        if end_ov_oob > 0:
            end_ov -= end_ov_oob
        end_rel = start_rel + tile_size
        if i == n_tiles - 1:
            end_rel += size % n_tiles

        starts_ov.append(start_ov)
        ends_ov.append(end_ov)
        starts_rel.append(start_rel)
        ends_rel.append(end_rel)

    return starts, starts_rel, starts_ov, ends, ends_rel, ends_ov


def extract_image_tiles(img: numpy.ndarray, 
                        x_props: tuple[int, int, int, int, int, int],
                        y_props: tuple[int, int, int, int, int, int]) -> list['ImageTile']:
    """Extracts image tile using tiling indices for x and y axes. 

    Args:
        img (numpy.ndarray): Image to be tiled.
        x_props (tuple[int, int, int, int, int, int]): Tiling information in x-axis.
        y_props (tuple[int, int, int, int, int, int]): Tiling information in y-axis.

    Returns:
        list[ImageTile]: List of image tiles.
    """
    img_tiles = []
    for (x_start, x_start_rel, x_start_ov, x_end, x_end_rel, x_end_ov) in zip(*x_props):
        for (y_start, y_start_rel, y_start_ov, y_end, y_end_rel, y_end_ov) in zip(*y_props):
            img_ = img[x_start_ov:x_end_ov, y_start_ov:y_end_ov]
            x_props_ = (x_start, x_start_rel, x_start_ov, x_end, x_end_rel, x_end_ov)
            y_props_ = (y_start, y_start_rel, y_start_ov, y_end, y_end_rel, y_end_ov)
            img_tile = ImageTile(img_, x_props_, y_props_, img.shape[:2])
            img_tiles.append(img_tile)
    return img_tiles

        
def reassemble_np(image_tiles: list['ImageTile'], 
               outputshape: tuple[int, int] | tuple[int, int, int]) -> numpy.ndarray:
    """Reassembles image as numpy.ndarray tiles to an image.

    Args:
        image_tiles (list[ImageTile]): Tiles to reassemble.
        outputshape (tuple[int, int] | tuple[int, int, int]): Shape of the output image that is reassembled.

    Returns:
        numpy.ndarray: Reassembled image.
    """
    template = np.zeros(outputshape)
    for idx, img_tile in enumerate(image_tiles):
        start_x = img_tile.x_props[0]
        start_x_rel = img_tile.x_props[1]
        end_x = img_tile.x_props[3]
        end_x_rel = img_tile.x_props[4]
        start_y = img_tile.y_props[0]
        start_y_rel = img_tile.y_props[1]
        end_y = img_tile.y_props[3]
        end_y_rel = img_tile.y_props[4]
        img = img_tile.image
        template[start_x:end_x, start_y:end_y] = img[start_x_rel:end_x_rel,start_y_rel:end_y_rel]
    return template
    

def reassemble_sitk_displacement_field(image_tiles: list['ImageTile'], 
               outputshape: tuple[int, int] | tuple[int, int, int]) -> numpy.ndarray:
    """Reassembles tile-wise displacement fields into a whole image displacement field.

    Args:
        image_tiles (list[&#39;ImageTile&#39;]): _description_
        outputshape (tuple[int, int] | tuple[int, int, int]): _description_

    Returns:
        numpy.ndarray: Displacement field as numpy.ndarray. 
    """
    template = np.zeros((outputshape[0], outputshape[1], 2))
    for idx, img_tile in enumerate(image_tiles):
        start_x = img_tile.x_props[0]
        start_x_rel = img_tile.x_props[1]
        end_x = img_tile.x_props[3]
        end_x_rel = img_tile.x_props[4]
        start_y = img_tile.y_props[0]
        start_y_rel = img_tile.y_props[1]
        end_y = img_tile.y_props[3]
        end_y_rel = img_tile.y_props[4]
        
        displ = img_tile.image
        displ = displ[start_y_rel:end_y_rel, start_x_rel:end_x_rel]
        displ_np = sitk.GetArrayFromImage(displ)
        template[start_x:end_x, start_y:end_y] = displ_np
    return template


# def extract_image_tiles_from_tile_sizes(img: numpy.ndarray,
#                                         x_props: tuple[\
#                                             numpy.ndarray, \
#                                             numpy.ndarray, \
#                                             numpy.ndarray, \
#                                             numpy.ndarray
#                                             ],
#                                         y_props: tuple[\
#                                             numpy.ndarray, \
#                                             numpy.ndarray, \
#                                             numpy.ndarray, \
#                                             numpy.ndarray
#                                             ]):
#     img_tiles = []
#     for (x_start, x_start_rel, x_end, x_end_rel) in zip(*x_props):
#         for (y_start, y_start_rel, y_end, y_end_rel) in zip(*y_props):
#             img_ = img[x_start:x_end, y_start:y_end] 
#             x_tile_props_ = (x_start, x_start_rel, x_end, x_end_rel)
#             y_tile_props_ = (y_start, y_start_rel, y_end, y_end_rel)
#             img_tile = ImageTile(img_, x_tile_props_, y_tile_props_, img.shape[:2])
#             img_tiles.append(img_tile)
#     return img_tiles


def reassemble_np_from_tile_size(image_tiles: list[ImageTile],
                                 outputshape: tuple[int, int] | tuple[int, int, int]) \
                                     -> numpy.ndarray:
    template = np.zeros(outputshape)
    for idx, img_tile in enumerate(image_tiles):
        start_x = img_tile.x_props[0]
        end_x = img_tile.x_props[2]
        start_y = img_tile.y_props[0]
        end_y = img_tile.y_props[2]
        img = img_tile.image        
        template[start_x:end_x, start_y:end_y] = img
    return template


def reassemble_sitk_displacement_field_from_tile_size(image_tiles: list['ImageTile'], 
               outputshape: tuple[int, int] | tuple[int, int, int]) -> numpy.ndarray:
    """Reassembles tile-wise displacement fields into a whole image displacement field.

    Args:
        image_tiles (list[&#39;ImageTile&#39;]): _description_
        outputshape (tuple[int, int] | tuple[int, int, int]): _description_

    Returns:
        numpy.ndarray: Displacement field as numpy.ndarray. 
    """
    template = np.zeros((outputshape[0], outputshape[1], 2))
    for idx, img_tile in enumerate(image_tiles):
        start_x = img_tile.x_props[0]
        end_x = img_tile.x_props[3]
        start_y = img_tile.y_props[0]
        end_y = img_tile.y_props[3]
        start_int_x = img_tile.x_props[1]
        end_int_x = img_tile.x_props[4]
        start_int_y = img_tile.y_props[1]
        end_int_y = img_tile.y_props[4]
        start_rel_x = img_tile.x_props[2]
        end_rel_x = img_tile.x_props[5]
        start_rel_y = img_tile.y_props[2]
        end_rel_y = img_tile.y_props[5]
        displ = img_tile.image
        displ_np = sitk.GetArrayFromImage(displ)
        template[start_rel_x:end_rel_x, start_rel_y:end_rel_y] = displ_np[start_int_x:end_int_x, start_int_y:end_int_y]
        # Change from 0, 2 to 1, 3
        # start_x = img_tile.x_props[1]
        # end_x = img_tile.x_props[3]
        # start_y = img_tile.y_props[1]
        # end_y = img_tile.y_props[3]
        # displ = img_tile.image
        # displ_np = sitk.GetArrayFromImage(displ)
        # template[start_x:end_x, start_y:end_y] = displ_np[start_x:end_x, start_y:end_y]
    return template


# def get_tile_params_by_tile_size(size: int, 
#                                  tile_size: int, 
#                                  min_overlap: float = 0) \
#                                      -> tuple[\
#                                          numpy.ndarray, \
#                                          numpy.ndarray, \
#                                          numpy.ndarray, \
#                                          numpy.ndarray, \
#                                          numpy.ndarray, \
#                                          numpy.ndarray \
#                                          ]:
#     overlaps = []
#     start = 0
#     min_overlap_px = int(tile_size * min_overlap)
#     interval_step = tile_size - min_overlap_px
#     starts = []
#     starts_rel = []
#     starts_int = []
#     ends = []
#     ends_rel = []    
#     ends_int = []
#     min_overlap_px = min_overlap_px // 2
#     while start + tile_size < size:
#         starts.append(start)
#         # Relative start/end points within tile.
#         if start == 0:
#             start_rel = 0
#             start_int = 0
#         else:
#             start_int = min_overlap_px
#             start_rel = start + min_overlap_px
#         end = start + tile_size
#         end_rel = end - min_overlap_px 
#         end_int = tile_size - min_overlap_px
#         starts_int.append(start_int)
#         starts_rel.append(start_rel)
#         ends.append(end)
#         ends_int.append(end_int)
#         ends_rel.append(end_rel)
#         start = start + interval_step
#         overlaps.append(min_overlap_px)
#     # TODO: For the last tile the overlap should be a split in the middle between the two images.
#     # TODO: At the moment the second to last tile will be overwritten with the overlapping area. 
#     if start + tile_size >= size:
#         start_prev = starts[-1]
#         start_rel = start_prev + tile_size - min_overlap_px
#         start = size - tile_size
#         start_int = tile_size + min_overlap_px
#         starts.append(start)
#         starts_rel.append(start_rel) 
#         end_rel = size
#         end = size
#         end_int = tile_size
#         ends.append(end)
#         ends_int.append(end_int)
#         ends_rel.append(end_rel)
#     starts = np.array(starts)
#     ends = np.array(ends)
#     starts_rel = np.array(starts_rel)
#     ends_rel = np.array(ends_rel)
#     starts_int = np.array(starts_int)
#     ends_int = np.array(ends_int)
#     return starts, starts_rel, starts_int, ends, ends_rel, ends_int

def get_tile_params_by_tile_size(size: int, 
                                 tile_size: int, 
                                 min_overlap: float = 0) \
                                     -> tuple[\
                                         numpy.ndarray, \
                                         numpy.ndarray, \
                                         numpy.ndarray, \
                                         numpy.ndarray, \
                                         numpy.ndarray, \
                                         numpy.ndarray \
                                         ]:
    overlaps = []
    start = 0
    min_overlap_px = int(tile_size * min_overlap)
    interval_step = tile_size - min_overlap_px
    starts = []
    starts_rel = []
    starts_int = []
    ends = []
    ends_rel = []    
    ends_int = []
    min_overlap_px = min_overlap_px // 2
    if size <= tile_size:
        starts.append(0)
        starts_rel.append(0)
        starts_int.append(0)
        ends.append(size)
        ends_rel.append(size)
        ends_int.append(size)
        return starts, starts_int, starts_rel, ends, ends_int, ends_rel
    while start + tile_size < size:
        starts.append(start)
        # Relative start/end points within tile.
        if start == 0:
            start_rel = 0
            start_int = 0
        else:
            start_int = min_overlap_px
            start_rel = start + min_overlap_px
        end = start + tile_size
        end_rel = end - min_overlap_px 
        end_int = tile_size - min_overlap_px
        starts_int.append(start_int)
        starts_rel.append(start_rel)
        ends.append(end)
        ends_int.append(end_int)
        ends_rel.append(end_rel)
        start = start + interval_step
        overlaps.append(min_overlap_px)
    # TODO: For the last tile the overlap should be a split in the middle between the two images.
    # TODO: At the moment the second to last tile will be overwritten with the overlapping area. 
    if start + tile_size >= size:
        start_prev = starts[-1]
        # start_rel = start_prev + tile_size - min_overlap_px
        # start = size - tile_size
        # start_int = min_overlap_px
        end_rel = size
        end = size
        end_int = tile_size
        # start_rel = start_prev + tile_size - min_overlap_px
        start_rel = size - tile_size + min_overlap_px
        start = size - tile_size
        start_int = min_overlap_px
        starts.append(start)
        starts_int.append(start_int)
        starts_rel.append(start_rel) 
        ends.append(end)
        ends_int.append(end_int)
        ends_rel.append(end_rel)
    starts = np.array(starts)
    ends = np.array(ends)
    starts_rel = np.array(starts_rel)
    ends_rel = np.array(ends_rel)
    starts_int = np.array(starts_int)
    ends_int = np.array(ends_int)
    return starts, starts_int, starts_rel, ends, ends_int, ends_rel


def extract_image_tiles_from_tile_sizes(img: numpy.ndarray,
                                        x_props: tuple[\
                                            numpy.ndarray, \
                                            numpy.ndarray, \
                                            numpy.ndarray, \
                                            numpy.ndarray
                                            ],
                                        y_props: tuple[\
                                            numpy.ndarray, \
                                            numpy.ndarray, \
                                            numpy.ndarray, \
                                            numpy.ndarray
                                            ]):
    img_tiles = []
    for (x_start, x_start_int, x_start_rel, x_end, x_end_int, x_end_rel) in zip(*x_props):
        for (y_start, y_start_int, y_start_rel, y_end, y_end_int, y_end_rel) in zip(*y_props):
            img_ = img[x_start:x_end, y_start:y_end] 
            x_tile_props_ = (x_start, x_start_int, x_start_rel, x_end, x_end_int, x_end_rel)
            y_tile_props_ = (y_start, y_start_int, y_start_rel, y_end, y_end_int, y_end_rel)
            img_tile = ImageTile(img_, x_tile_props_, y_tile_props_, img.shape[:2])
            img_tiles.append(img_tile)
    return img_tiles