from hardware_utils import OmniNgramsTemp

spaces = {}

matchers['stg_temp_max'] = OmniNgramsTemp(n_max=2)

def get_space(attr):
    return matchers[attr]