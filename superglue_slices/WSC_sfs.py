from .general_sfs import slice_func_dict as general_slice_func_dict

slices = []

slice_func_dict = {slice.name: slice for slice in slices}
slice_func_dict.update(general_slice_func_dict)