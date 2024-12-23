from dataclasses import dataclass
import numpy, numpy as np

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

    start = 0
    starts = []
    starts_ov = []
    ends = []
    ends_ov = []
    starts_rel = []
    ends_rel = []
    starts = [0]
    starts_rel = [0]
    starts_ov = [0]
    ends = [starts[0] + tile_size]
    ends_rel = [starts_rel[0] + tile_size]
    ends_ov = [min(ends[0] + overlap, size)]
    for i in range(1, n_tiles - 1):
        start = tile_size * i
        start_ov = max(int(start - overlap_offset), 0)
        end = start + tile_size
        end_ov = min(int(end + overlap_offset), size)
        start_rel = overlap_offset
        end_rel = tile_size + overlap_offset
        starts.append(start)
        starts_rel.append(start_rel)
        starts_ov.append(start_ov)
        ends.append(end)
        ends_rel.append(end_rel)
        ends_ov.append(end_ov)
    ends.append(size)
    ends_rel.append(overlap + tile_size + size & n_tiles)
    ends_ov.append(size)

    # ends[-1] = size
    # ends_ov[-1] = size
    # ends_rel[-1] += size % n_tiles
    return starts, starts_rel, starts_ov, ends, ends_rel, ends_ov


@dataclass
class ImageTile:
    """ImageTile class that contains an image tile and indices that were used to extract the
    tile and that are necessary to locate the image tile in its original source image.
    """
    
    image: numpy.ndarray
    x_props: tuple[int, int, int, int, int, int]
    y_props: tuple[int, int, int, int, int, int]


def extract_image_tiles(img: numpy.ndarray, 
                        x_props: tuple[int, int, int, int, int, int],
                        y_props: tuple[int, int, int, int, int, int]) -> list['ImageTile']:
    """Extracts image tile using tiling indices for x and y axes. 

    Returns:
        list[ImageTile]: List of image tiles.
    """
    img_tiles = []
    for (x_start, x_start_rel, x_start_ov, x_end, x_end_rel, x_end_ov) in zip(*x_props):
        for (y_start, y_start_rel, y_start_ov, y_end, y_end_rel, y_end_ov) in zip(*y_props):
            img_ = img[x_start_ov:x_end_ov, y_start_ov:y_end_ov]
            x_props_ = (x_start, x_start_rel, x_start_ov, x_end, x_end_rel, x_end_ov)
            y_props_ = (y_start, y_start_rel, y_start_ov, y_end, y_end_rel, y_end_ov)
            img_tile = ImageTile(img_, x_props_, y_props_)
            img_tiles.append(img_tile)
    return img_tiles

    
def reassemble(image_tiles: list['ImageTile'], 
               outputshape: tuple[int, int] | tuple[int, int, int]) -> numpy.ndarray:
    """Reassembles image tiles to an image.

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
    