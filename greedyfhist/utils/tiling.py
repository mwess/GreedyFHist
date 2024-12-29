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
    

def get_tile_params(size: int, n_tiles: int = 2, overlap: float = 0.1) -> tuple[int, int, int, int, int, int]:
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