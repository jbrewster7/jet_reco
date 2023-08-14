import awkward as ak

def ak_equals(a1,a2):
    '''
    returns boolean array in the shape of a1 where each true represents an element that was found in anywhere 
    in the same level in a2 
    
    a1 and a2 must have the same number of dimensions
    '''
    arg_cart = ak.argcartesian({'a1':a1,'a2':a2},axis=-1) 
    cart_mask = a1[arg_cart['a1']] == a2[arg_cart['a2']] # compare the two arrays using cartesian product
    
    # reshape into the shape of a1 set an element to true if that element was found anywhere in the equivalent level in a2
    return ak.any(ak.unflatten(cart_mask,ak.flatten(ak.run_lengths(arg_cart['a1'])),axis=-1),axis=-1)