import numpy, numpy as np
import pandas


def eucl(src: numpy.ndarray, dst: numpy.ndarray) -> float:
    """Computes Euclidean distance between src and dst array.

    Args:
        src (numpy.ndarray):
        dst (numpy.ndarray):

    Returns:
        float:
    """
    return np.sqrt(np.square(src[:, 0] - dst[:, 0]) + np.square(src[:, 1] - dst[:, 1]))


def compute_distance_for_lm(warped_df: pandas.DataFrame, fixed_df: pandas.DataFrame) -> pandas.DataFrame:
    """Compute target registration error for each pair of matching landmarks. Landmarks are matched with the 'label' column.

    Args:
        warped_df (pandas.DataFrame):
        fixed_df (pandas.DataFrame):

    Returns:
        pandas.DataFrame: 
    """
    merged_df = warped_df.merge(fixed_df, on='label', suffixes=('_src', '_dst'))
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df.dropna(inplace=True)
    src_mat = merged_df[['x_src', 'y_src']].to_numpy()
    dst_mat = merged_df[['x_dst', 'y_dst']].to_numpy()
    merged_df['tre'] = eucl(src_mat, dst_mat)
    return merged_df


def compute_tre(target_landmarks: pandas.DataFrame, 
                warped_landmarks: pandas.DataFrame, 
                target_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Computes target registration error based metrics: 
        - mean relative target registration error
        - median relative target registration error
        - mean target registration error
        - median target registration error
    

    Args:
        target_landmarks (pandas.DataFrame): 
        warped_landmarks (pandas.DataFrame): 
        target_shape (tuple[int, int]): Shape of target image. Used for computing relative target registration error.

    Returns:
        tuple[float, float, float, float]: mean_rTRE, median_rTRE, mean_TRE, median_TRE
    """
    unified_lms = compute_distance_for_lm(warped_landmarks, target_landmarks)
    image_diagonal = np.sqrt(np.square(target_shape[0]) + np.square(target_shape[1]))
    unified_lms['rtre'] = unified_lms['tre']/image_diagonal
    mean_rtre = np.mean(unified_lms['rtre'])
    median_rtre = np.median(unified_lms['rtre'])
    median_tre = np.median(unified_lms['tre'])
    mean_tre = np.mean(unified_lms['tre'])
    return mean_rtre, median_rtre, mean_tre, median_tre
