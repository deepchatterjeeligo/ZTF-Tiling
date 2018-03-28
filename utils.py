#################################################
### Utility scripts/ functions for ZTF_Tiling ###
#################################################
import numpy as np
import pandas as pd
from os import path

def removeTileFromSkymap(skymap, tile_content, \
                                return_flag=True, tol=0.05):
    '''
    Function        :: Removes RAs and Decs in tile_content from skymaps.
                       Implementation based on numpy.in1d
    
    skymap          :: Skymap in form of [ra, dec, pval]
    
    tile_content    :: Contents of a tile in the form of 
                       [ra_tile, dec_tile, pval_tile]
    
    return_flag     :: If supplied returns True if probability
                       contained in tile differs from probability
                       removed from skymap by the tol amount
    tol             :: Tolerance in difference of probability value
                       between that contained in tile and that removed
                       from skymap
    '''
    [ra, dec, pval] = skymap
    [ra_tile, dec_tile, pval_tile]  = tile_content
    
    ra_mask     = np.in1d(ra, ra_tile)
    dec_mask    = np.in1d(dec, dec_tile)
    
    net_mask    = np.logical_not( ra_mask * dec_mask )
    
    # Filter the RAs, Decs, pVals
    ra_new      = ra[net_mask]
    dec_new     = dec[net_mask]
    pval_new    = pval[net_mask]
    
    # Sanity checker
    if return_flag:
        diff =  np.abs( np.sum(pval[~net_mask]) - np.sum(pval_tile) )
        return [ra_new, dec_new, pval_new], diff < tol
    return [ra_new, dec_new, pval_new]
    
    
def removeTileFromSkymapv2(skymap, tile_content, \
                                return_flag=True, tol=1e-10):
    '''
    Function        :: Similar to removeTileFromSkymap, differs
                       in the definition of tol
                       Implementation based on numpy.in1d and broadcasting
    
    skymap          :: Skymap in form of [ra, dec, pval]
    
    tile_content    :: Contents of a tile in the form of 
                       [ra_tile, dec_tile, pval_tile]
    
    tol             :: Difference in between arrays (if possible after
                       broadcasting)
    '''
    def in1d_alternate(a, b , tol):
        S = round(1/tol)
        return np.in1d(np.around(a*S).astype(int),\
                                np.around(b*S).astype(int))
    [ra, dec, pval] = skymap
    [ra_tile, dec_tile, pval_tile]  = tile_content
    
    ra_mask     = in1d_alternate(ra, ra_tile, tol)
    dec_mask    = in1d_alternate(dec, dec_tile, tol)
    
    net_mask    = np.logical_not( ra_mask * dec_mask )
    
    # Filter the RAs, Decs, pVals
    ra_new      = ra[net_mask]
    dec_new     = dec[net_mask]
    pval_new    = pval[net_mask]
    
    return [ra_new, dec_new, pval_new]

def skymapToFits(skymap, filename): #FIXME: add header information later
    '''
    Function	:: Pack the skymap into a fits file using astropy fits
    
    skymap	:: Skymap in the form of [ra, dec, pvals]

    filename	:: Output filename. Intermediate directories are created
    '''
    from astropy.io import fits
    if path.dirname(filename) and not path.exists(path.dirname(filename)):
        os.makedirs(path.dirname(filename))
    hdu		= fits.primaryHDU(skymap)
    hdulist 	= fits.HDUList([hdu])
    
    hdulist.writeto(filename)
