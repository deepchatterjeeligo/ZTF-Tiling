# Copyright (C) 2017 Shaon Ghosh, David Kaplan, Shasvath Kapadia, Deep Chatterjee
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""

Creates ranked tiles for a given gravitational wave trigger. Sample steps are below:


tileObj = rankedTilesGenerator.RankedTileGenerator('bayestar.fits.gz')
[ranked_tile_index, ranked_tile_probs] = tileObj.getRankedTiles(resolution=512)

This gives the ranked tile indices and their probabilities for the bayestar sky-map.
The resolution is 512, thus ud_grading to this value from the actual sky-map resolution.
The code expects the file ZTF_tiles_set1_nowrap_indexed.dat and the pickled file 
preComputed_pixel_indices_512.dat to be in the same path. 

"""

import os
import sys
import time
import pickle
import datetime
import numpy as np
import pylab as pl
import pandas as pd
from math import ceil

import healpy as hp
import ConfigParser
#from scipy import interpolate
#from scipy.stats import norm

#from astropy.time import Time
#from astropy import units as u
#from astropy.table import Table
#from astropy.coordinates import get_sun
#from astropy.coordinates import get_moon
#from astropy.coordinates import get_body
#from astropy.utils.console import ProgressBar

#from astropy.coordinates import SkyCoord, EarthLocation, AltAz


def getTileBounds(FOV, ra_cent, dec_cent):
    dec_down = dec_cent - 0.5*np.sqrt(FOV)
    dec_up = dec_cent + 0.5*np.sqrt(FOV)

    ra_down_left = ra_cent - 0.5*(np.sqrt(FOV)/(np.cos(dec_down*(np.pi/180.))))
    ra_down_right = ra_cent + 0.5*(np.sqrt(FOV)/(np.cos(dec_down*(np.pi/180.))))
    ra_up_left = ra_cent - 0.5*(np.sqrt(FOV)/(np.cos(dec_up*(np.pi/180.))))
    ra_up_right = ra_cent + 0.5*(np.sqrt(FOV)/(np.cos(dec_up*(np.pi/180.))))
    
    return([dec_down, dec_up, ra_down_left, ra_down_right, ra_up_left, ra_up_right])


class RankedTileGenerator:
		
	def __init__(self, skymapfile, skymap3D=False):
		'''
		skymapfile :: The GW sky-localization map for the event
		path	   :: Path to the preCoputed files
		preComputeFiles  :: A list of all the precompute files
		skymap3D	:: Whether the sky-map is 3D or not (default: False)
		'''
		

		self.preComputed_256 = 'preComputed_ZTF_pixel_indices_256.dat'


		if skymap3D:
			self.skymap3D = skymap3D
			self.skymap, self.distmu,\
			self.distsigma, self.distnorm = hp.read_map(skymapfile, field=range(4), verbose=False)
		else:
			self.skymap = hp.read_map(skymapfile, verbose=False)

		npix = len(self.skymap)
		self.nside = hp.npix2nside(npix)
		tileFile = 'ZTF_tiles_indexed.dat'
		self.tileData = np.recfromtxt(tileFile, names=True)
		
	def getRankedTiles(self, verbose=False):
		'''
		METHOD		:: This method returns two numpy arrays, the first
				   contains the tile indices of telescope and the second
				   contains the probability values of the corresponding 
				   tiles. The tiles are sorted based on their probability 
				   values.
		
		resolution  :: The value of the nside, if not supplied, 
			       the default skymap is used.
		'''
		resolution = 256
		filename = self.preComputed_256

		File = open(filename, 'rb')
			
		data = pickle.load(File)
		tile_index = np.arange(len(data))
		skymapUD = hp.ud_grade(self.skymap, resolution, power=-2)
		npix = len(skymapUD)
		theta, phi = hp.pix2ang(resolution, np.arange(0, npix))
		pVal = skymapUD[np.arange(0, npix)]

		allTiles_probs = []
		for ii in range(0, len(data)):
			pTile = np.sum(pVal[data[ii]])
			allTiles_probs.append(pTile)


		allTiles_probs = np.array(allTiles_probs)
		index = np.argsort(-allTiles_probs)

		allTiles_probs_sorted = allTiles_probs[index]
		tile_index_sorted = tile_index[index]

		#if self.skymap3D:
		#	for ii in np.arange(len(data))[index]:
		#		meanDist = self.getDistEstInTile(data[ii])
		
		return [tile_index_sorted, allTiles_probs_sorted]



	def createTilePriorityList(self, CI=0.9, T_dur=2*3600.0, T_int_up=1200., T_int_low=60.):
		'''
		Method  :: This method outputs the priority list of tiles with 
				   tile IDs, RA, Dec, exposure times in a csv format.
				   The exposure time is calculated based on the T_duration
				   which is the total duration for observation. We used 
				   eq. 2 of arXiv:1708.06723 to comute this.
		'''
		[tile_index_sorted, allTiles_probs_sorted] = self.getRankedTiles()
		include = np.cumsum(allTiles_probs_sorted) < CI
		include[np.sum(include)] = True
		tile_indices_CI = tile_index_sorted[include]
		allTiles_probs_CI = allTiles_probs_sorted[include]
		t_int = (allTiles_probs_CI/np.sum(allTiles_probs_CI))*T_dur
		t_int[t_int > T_int_up] = T_int_up
		t_int[t_int < T_int_low] = T_int_low
		ra_center = self.tileData['ra_center'][np.isin(self.tileData['ID'], tile_indices_CI)]
		dec_center = self.tileData['dec_center'][np.isin(self.tileData['ID'], tile_indices_CI)]
		program_id = 2*np.ones(len(tile_indices_CI), dtype='int')
		filter_id = np.ones(len(tile_indices_CI), dtype='int')
		subprogram_name = ['TOO_EMGW']*len(tile_indices_CI)
		columns = [ 'program_id', 'field_id', 'filter_id', 'ra',\
					'dec', 'exposure_time', 'subprogram_name']
		df = pd.DataFrame(np.vstack((program_id, tile_indices_CI,\
									 filter_id, ra_center, dec_center,\
									 np.round(t_int, 0), subprogram_name)).T,\
									 columns=columns)
#		df = pd.DataFrame(np.vstack((program_id, tile_indices_CI,\
#									 filter_id, ra_center, dec_center,\
#									 np.round(t_int, 0), subprogram_name)).T,\
#									 columns=None)

		df.to_csv('priority.csv', index=False)
		return df







