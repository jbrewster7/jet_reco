import awkward as ak

def ak_equals(a1,a2,axis=-1):
    '''
    returns boolean array in the shape of a1 where each true represents an element that was found in anywhere 
    in the same level in a2 
    '''
    arg_cart = ak.argcartesian({'a1':a1,'a2':a2},axis=axis)
    cart_mask = a1[arg_cart['a1']] == a2[arg_cart['a2']]
    return ak.any(ak.unflatten(cart_mask,ak.flatten(ak.run_lengths(arg_cart['a1'])),axis=axis),axis=axis)