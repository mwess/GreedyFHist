from dataclasses import dataclass

import numpy

from greedyfhist.options import RegistrationOptions
from greedyfhist.registration import GreedyFHist, RegistrationResult
from greedyfhist.utils.io import create_if_not_exists


@dataclass
class HistologySection:
    """Composite object for keeping imaging data from the same section together.

    Attributes
    ----------
    
    ref_image: Optional[Any]
        Image used for registration.
        
    ref_mask: Optional[Any]
        Optional mask used for registration.
        
    additional_data: List[Any]
        Contains additional imaging data.
    """
    
    ref_image: Optional[Any]
    ref_mask: Optional[Any]
    additional_data: List[Any]
    
    def register_to_image(self, 
                          fixed_image: numpy.array, 
                          fixed_mask: Optional[numpy.array] = None, 
                          options: Optional[RegistrationOptions] = None,
                          registerer: Optional[GreedyFHist] = None) -> RegistrationResult:
        if registerer is None:
            registerer = GreedyFHist.load_from_config({})

        registration_result = registerer.register(
            self.ref_image.data,
            fixed_image.data,
            self.ref_mask,
            fixed_mask,
            options=options
        )
        return registration_result

    def apply_transformation(self,
                             registration_result: RegistrationResult,
                             registerer: Optional[GreedyFHist] = None) -> 'HistologySection':
        if registerer is None:
            registerer = GreedyFHist.load_from_config({})
        if self.ref_image is not None:
            warped_ref_image = registerer.transform_image(self.ref_image, registration_result, 'LINEAR')
        else:
            warped_ref_image = None
        if self.ref_mask is not None:
            warped_ref_mask = registerer.transform_image(self.ref_mask, registration_result, 'NN')
        else:
            warped_ref_mask = None
        warped_additional_data = []
        for ad in self.additional_data:
            warped_ad = ad.transform_data(registerer, registration_result)
            warped_additional_data.append(warped_ad)
        return HistologySection(ref_image=warped_ref_image,
                                ref_mask=warped_ref_mask,
                                additional_data=warped_additional_data)

    def to_directory(self, 
                     directory: str):
        create_if_not_exists(directory)
        if self.ref_image is not None:
            self.ref_image.to_directory(directory)
        if self.ref_mask is not None:
            self.ref_mask.to_directory(directory)
        for ad in self.additional_data:
            ad.to_directory(directory)
            