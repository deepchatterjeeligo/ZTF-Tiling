# from __future__ import division, print_function

import os
import sys
import numpy as np
import pandas as pd
import configparser
import pickle

from astropy.utils.console import ProgressBar

import getFermiSkyMap



def getArea(a, b, c):
    s = 0.5*(a + b + c)
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
    return area

def getCCDBounds(ra_cent, dec_cent, CCD_dim=1.707):
    '''
    For a choice of ra center and dec center and the dimension of the CCD
    this function gives the ra and dec for the corners of the FOV.
    NOTE: This function can be also used to compute the Tile boundaries
    '''
    dec_down = dec_cent - 0.5*CCD_dim
    dec_up = dec_cent + 0.5*CCD_dim

    ra_down_left = ra_cent - 0.5*(CCD_dim/(np.cos(dec_down*(np.pi/180.))))
    ra_down_right = ra_cent + 0.5*(CCD_dim/(np.cos(dec_down*(np.pi/180.))))
    ra_up_left = ra_cent - 0.5*(CCD_dim/(np.cos(dec_up*(np.pi/180.))))
    ra_up_right = ra_cent + 0.5*(CCD_dim/(np.cos(dec_up*(np.pi/180.))))
    
    return([dec_down, dec_up, ra_down_left, ra_down_right, ra_up_left, ra_up_right])

def getCCDcenters(ra_tile_center, dec_tile_center):
	'''
	From Michael Coughlin:
	Computes the centers of the CCD for a given tile of observation
	'''
	pixel_size = 15 # microns
	plate_scale = 1 # arcsec/pixel
	plate_scale_deg = plate_scale / 3600.0
	tile_size = [6144,6160] # pixels
	gap_size = np.array([7,10])/(1e-3*pixel_size) # pixels
	
	idxs = [-3/2, -1/2, 1/2, 3/2]
# 	CCD = np.zeros((len(idxs),len(idxs),2))
	CCD = np.array([])
	for i,idx in enumerate(idxs):
		for j,idy in enumerate(idxs):
			ra_center = ra_tile_center + idx*(tile_size[0]+gap_size[0])*plate_scale_deg
			dec_center = dec_tile_center + idy*(tile_size[1]+gap_size[1])*plate_scale_deg
# 			CCD[i,j,0] = ra_center
# 			CCD[i,j,1] = dec_center
			CCD = np.append(CCD, np.array([ra_center, dec_center]))
	return CCD.reshape(int(len(CCD.flatten())/2), 2)


def getFourTriangles(point_ra, point_dec, row):
	'''
	Function	:: Given a point and edges of the tiles, this function computes the area
				   of the four consecutive triangles formed by the point and the edges.
				   **************
				   * *		  *	*
				   *   *	*	*
				   *	 *		*
				   *   *	*   *
				   * *		  *	*
				   **************
				   
	'''
	triangle1 = [((point_ra - row['ra_Left_U'])**2 + (point_dec - row['Dec_Up'])**2)**0.5,\
				((point_ra - row['ra_Right_U'])**2 + (point_dec - row['Dec_Up'])**2)**0.5,\
				np.abs(row['ra_Left_U'] - row['ra_Right_U'])]

	triangle2 = [((point_ra - row['ra_Left_U'])**2 + (point_dec - row['Dec_Up'])**2)**0.5,\
				((point_ra - row['ra_Left_D'])**2 + (point_dec - row['Dec_Down'])**2)**0.5,\
				((row['ra_Left_U'] - row['ra_Left_D'])**2 + (row['Dec_Up'] - row['Dec_Down'])**2)**0.5]

	triangle3 = [((point_ra - row['ra_Left_D'])**2 + (point_dec - row['Dec_Down'])**2)**0.5,\
				((point_ra - row['ra_Right_D'])**2 + (point_dec - row['Dec_Down'])**2)**0.5,\
				np.abs(row['ra_Left_D'] - row['ra_Right_D'])]

	triangle4 = [((point_ra - row['ra_Right_D'])**2 + (point_dec - row['Dec_Down'])**2)**0.5,\
				((point_ra - row['ra_Right_U'])**2 + (point_dec - row['Dec_Up'])**2)**0.5,\
				((row['ra_Right_U'] - row['ra_Right_D'])**2 + (row['Dec_Up'] - row['Dec_Down'])**2)**0.5]

	area1 = getArea(triangle1[0], triangle1[1], triangle1[2])
	area2 = getArea(triangle2[0], triangle2[1], triangle2[2])
	area3 = getArea(triangle3[0], triangle3[1], triangle3[2])
	area4 = getArea(triangle4[0], triangle4[1], triangle4[2])

	return area1 + area2 + area3 + area4


class RankedTileGenerator:
	'''
	Class	:: This class instantiates a ranked tile generator object. The instantiation
			   needs a config file that specifies the web address of the fits, the year 
			   of the GRB and the path to the ZTF tiles file
	'''
	def __init__(self, configDict, skymap):
# 		config = configparser.ConfigParser()
# 		config.read(configfile)
# 		self.webaddress = config.get('Paths', 'webaddress')
# 		self.year = config.get('Paths', 'year')
		tileFile = configDict['tileFile']
		self.skymap = skymap
		if 'catalog' in configDict.keys():
			self.galaxy_catalog = configDict['catalog']
			
			if self.galaxy_catalog:
				assert ((self.galaxy_catalog=='GLADE') or (self.galaxy_catalog=='CLU')\
					   or (self.galaxy_catalog=='GWGC')), "Currently only allowed galaxy catalogs are 'GLADE', 'CLU' and 'GWGC' "
# 			except AssertionError:
# 				print("Currently only allowed galaxy catalogs are 'GLADE', 'CLU' and 'GWGC' ")
# 				sys.exit(0)
		self.tileData = pd.DataFrame(np.recfromtxt(tileFile, names=True))

		### Output directories ###
# 		self.fitsfileDir = config.get('Output', 'fitsfileDir')
		self.rankedTilesDir = configDict['rankedTilesDir']
		

	def getGalaxiesInTile(self, ID):
		if self.galaxy_catalog == 'GLADE':
			File = open('GLADECatalog_database/Galaxies_in_tile_' + str(ID) + '.dat', 'rb')
		if self.galaxy_catalog == 'CLU':
			File = open('CLUCatalog_database/Galaxies_in_tile_' + str(ID) + '.dat', 'rb')
		if self.galaxy_catalog == 'GWGC':
			File = open('GWGCCatalog_database/Galaxies_in_tile_' + str(ID) + '.dat', 'rb')

		data = pickle.load(File)
		File.close()
		return data


	def getRankedTiles(self, hp=False, multiCCD=True):
		'''
		Method 	  :: For a Fermi event which have the localization region in fits 
					 format, this method finds the list of the ranked tiles. The 
					 ranked tiles are output in as an extra column to the tile data
					 obtained from the tiles file. If the fits file could no be 
					 found from the link, then the extra column will be filled with
					 nans.
					 
		event	  :: Name of the Fermi event, e.g. "bn180116678". The path to this event
					 should be specified in the config file. In the following format:
					 [Paths]
  					 webaddress = https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/
					 year = 2018
		hp		  :: Set it to True to get ranked-tiles for sky-localization maps in HEALPix
		
		
		Output	  :: If the config file does not have a catalog field in the Input section
					 with a valid entry name (GLADE, CLU or GWGC) then, this method returns
					 a pandas DataFrame with ranked list of tiles.
					 If the config file has a catalog field with a valid name of galaxy
					 catalog, then the output is a list, whose first element is a pandas
					 DataFrame with the ranked list of tiles, and the second element is 
					 a list of pandas DataFrames each of which are the list of galaxies
					 that are within the ranked-tiles. 
					 
		TODO	  :: Another argument that will use the galaxies that lie within the 
					 different tiles and some properties of these galaxies to find 
					 a weighted probability of the tiles to compute their weighted-ranks.
		'''
# 		if hp:
# 			fitsfilename = 'glg_healpix_all_' + event + '.fit'
# 		else:
# 			fitsfilename = 'glg_locprob_all_' + event + '.fit'
# 		f = os.path.join(self.webaddress, self.year, event, 'quicklook', fitsfilename)
# 		fermi_skymap = getFermiSkyMap.getFermiSkyMap(f, self.fitsfileDir, healpix=hp)
# 		if fermi_skymap is not None:
# 			(self.point_ra_all,\
# 			 self.point_dec_all,\
# 			 self.point_pVal_all) = (fermi_skymap[0],\
# 									 fermi_skymap[1],\
# 									 fermi_skymap[2])
# 									 
# 		else: # If the Fermi sky-map is not found the return nans for tile probabilities
# 			self.tileData['Tile_Value'] = np.ones(len(self.tileData))*np.nan
# 			return self.tileData

		
		(self.point_ra_all,\
		 self.point_dec_all,\
		 self.point_pVal_all) = (self.skymap[0],\
								 self.skymap[1],\
								 self.skymap[2])

		# Loop over tiles #
		pvalTile = []
# 		self.RATile = np.array([])
# 		self.DecTile = np.array([])
# 		self.PVals = np.array([])
		TileProbSum = 0.0
		print('Computing Ranked-Tiles...')
		with ProgressBar(len(self.tileData)) as bar:
			for index, row in self.tileData.iterrows():
				ra_min = np.min([row['ra_Left_U'], row['ra_Right_U'], row['ra_Left_D'], row['ra_Right_D']])
				ra_max = np.max([row['ra_Left_U'], row['ra_Right_U'], row['ra_Left_D'], row['ra_Right_D']])
					
				## Keeping part of the sky map that falls in the region in and around the tile ##
				keep = (row['Dec_Down'] <= self.point_dec_all)*\
					   (self.point_dec_all <= row['Dec_Up'])*\
					   (ra_min <= self.point_ra_all)*\
					   (ra_max >= self.point_ra_all)
		
				point_ra_kept = self.point_ra_all[keep]
				point_dec_kept = self.point_dec_all[keep]
				point_pVal_kept = self.point_pVal_all[keep]

				point_ra = point_ra_kept
				point_dec = point_dec_kept
				point_pVal = point_pVal_kept
	
				tile_area = getFourTriangles(row['ra_center'], row['dec_center'], row)
				alltriangles = getFourTriangles(point_ra, point_dec, row)
		
				if row['ra_Left_D'] < 0:
					wrapped_point_ra = point_ra - 360.0
					alltriangles_wrapped = getFourTriangles(wrapped_point_ra, point_dec, row)
			
				if row['ra_Right_D'] > 360:
					wrapped_point_ra = point_ra + 360.0
					alltriangles_wrapped = getFourTriangles(wrapped_point_ra, point_dec, row)

				inside = (np.round(alltriangles, 2) <= np.round(tile_area, 2))
				if (row['ra_Left_D'] < 0) or (row['ra_Right_D'] > 360):
					inside += (np.round(alltriangles_wrapped, 2) <= np.round(tile_area, 2))

				
				TileProbSum += np.sum(point_pVal[inside])
				
				if np.sum(inside) > 0: ## conduct further calculation only if tile contains points
					ra_intile = point_ra[inside]
					dec_intile = point_dec[inside]
					pVal_intile = point_pVal[inside]

					if multiCCD:
						'''
						Here the function for multi-CCD will be called. 
						This function will take as input the list of ra and decs and 
						The center of the tile and from there generate the CCDs and their 
						boundaries and return the points that fall inside the CCDs.
						Currently the way the code requires the knowledge of the tile corners
						to figure out which points are inside a quadrilateral FOV. However, in
						this case the corners will be computed for each CCD. So a new function
						needs to be written.
						'''
				
						CCD_centers = getCCDcenters(row['ra_center'], row['dec_center'])
						insideCCDs_thisTile = np.zeros(len(pVal_intile)).astype('bool')
						for thisCCD_center in CCD_centers:
							ccdrow = {}
							[ccdrow['Dec_Down'], ccdrow['Dec_Up'], ccdrow['ra_Left_D'],\
							ccdrow['ra_Right_D'], ccdrow['ra_Left_U'],\
							ccdrow['ra_Right_U']] = getCCDBounds(thisCCD_center[0], thisCCD_center[1])
					
							CCD_area = getFourTriangles(thisCCD_center[0], thisCCD_center[1], ccdrow)
							alltriangles = getFourTriangles(ra_intile, dec_intile, ccdrow)
		
							if ccdrow['ra_Left_D'] < 0:
								wrapped_ra_intile = ra_intile - 360.0
								alltriangles_wrapped = getFourTriangles(wrapped_ra_intile, dec_intile, ccdrow)
			
							if ccdrow['ra_Right_D'] > 360:
								wrapped_ra_intile = ra_intile + 360.0
								alltriangles_wrapped = getFourTriangles(wrapped_ra_intile, dec_intile, ccdrow)
					
							insideCCD = (np.round(alltriangles, 2) <= np.round(CCD_area, 2))
							if (ccdrow['ra_Left_D'] < 0) or (ccdrow['ra_Right_D'] > 360):
								insideCCD += (np.round(alltriangles_wrapped, 2) <= np.round(CCD_area, 2))

							insideCCDs_thisTile += insideCCD

					
						pvalTile.append(np.sum(pVal_intile[insideCCDs_thisTile]))
						self.RATile = ra_intile[insideCCDs_thisTile]
						self.DecTile = dec_intile[insideCCDs_thisTile]
						self.PVals = pVal_intile[insideCCDs_thisTile]

					else: ## multiCCD if-block
						self.RATile = ra_intile
						self.DecTile = dec_intile
						self.PVals = pVal_intile
						pvalTile.append(np.sum(self.PVals))

				else:
					pvalTile.append(0)
				bar.update()


		tileVals = np.array(pvalTile)
		self.pTiles = tileVals/TileProbSum
		self.tileData['Tile_Value'] = self.pTiles
		self.tileData = self.tileData.sort_values(by=['Tile_Value'], ascending=False)
		include = np.cumsum(self.tileData.Tile_Value.values) < 0.9
		if len(include) < len(self.tileData.Tile_Value.values):
			include[np.sum(include)] = True
		self.N90 = np.sum(include)

		if self.galaxy_catalog:
			print("Finding galaxies in tiles from the {} catalog".format(self.galaxy_catalog))
			self.galaxies_inThisTile = []
			for tileID in self.tileData['ID'].values:
				self.galaxies_inThisTile.append(self.getGalaxiesInTile(tileID))
			return [self.tileData, self.galaxies_inThisTile]
		try:
			os.system('mkdir -p ' + self.rankedTilesDir)
			try:
				ranked_tiles_filename = fitsfilename.split('.fit')[0] + '.xlsx'
				self.tileData.to_excel(os.path.join(self.rankedTilesDir, ranked_tiles_filename))
			except:
				ranked_tiles_filename = fitsfilename.split('.fit')[0] + '.csv'
				self.tileData.to_csv(os.path.join(self.rankedTilesDir, ranked_tiles_filename))

		except:
			print('Could not write ranked tile generation file into location')
		return self.tileData

		
	def plotTiles(self, Num=None, size=(10, 8), tileEdge=True, N=None, save=False, fontsize=16):
		'''
		METHOD	 :: This method plots the sky-map and tiles for a given event. The user can
				   choose the number of tiles to be plotted. The default number of tiles 
				   that is plotted is the tiles constituting the top 90% tiles.
				   
		Num		 :: Total number of the ranked tiles to be plotted (Default is 90% CI tiles)
		size 	 :: Size of the figure (default is (10, 8))
		tileEdge :: Plots the edges of the tiles (default = True). If set to false then
					only plots the tile center. 
		N        :: Total number of tanked tiles for which the enclosed galaxies will be 
					plotted
					
		fontsize :: The size of the font for the plot labels.
		'''

		import AllSkyMap_basic
		import pylab as pl

		if np.any(np.isnan(self.tileData['Tile_Value'] )): 
			print('Could not find any sky-map to plot!')
			return None
		pl.rcParams.update({'font.size': fontsize})
		pl.figure(figsize=size)
		m = AllSkyMap_basic.AllSkyMap(projection='hammer')

		RAP_map, DecP_map = m(self.point_ra_all, self.point_dec_all)
		m.drawparallels(np.arange(-90.,120.,20.), color='grey', 
						labels=[False,True,True,False], labelstyle='+/-')
		m.drawmeridians(np.arange(0.,420.,30.), color='grey')
		m.drawmapboundary(fill_color='white')
		lons = np.arange(-150,151,30)
		m.label_meridians(lons, fontsize=fontsize, vnudge=1, halign='left', hnudge=-1)
		prob = self.point_pVal_all/np.sum(self.point_pVal_all)
		m.scatter(RAP_map, DecP_map, 10, prob) 
		pl.colorbar(orientation="horizontal", pad=0.04)

		if self.galaxy_catalog:
			RA_Gals = np.array([])
			Dec_Gals = np.array([])
			for ii in range(0, N):
				if self.galaxy_catalog == 'GWGC':
					RA_Gals = np.append(RA_Gals, self.galaxies_inThisTile[ii].RA.values)
					Dec_Gals = np.append(Dec_Gals, self.galaxies_inThisTile[ii].Dec.values)
				if self.galaxy_catalog == 'GLADE':
					RA_Gals = np.append(RA_Gals, self.galaxies_inThisTile[ii].RA.values)
					Dec_Gals = np.append(Dec_Gals, self.galaxies_inThisTile[ii].dec.values)
				if self.galaxy_catalog == 'CLU':
					RA_Gals = np.append(RA_Gals, self.galaxies_inThisTile[ii].ra.values)
					Dec_Gals = np.append(Dec_Gals, self.galaxies_inThisTile[ii].dec.values)

			RAP_Gals, DecP_Gals = m(RA_Gals, Dec_Gals)
			m.scatter(RAP_Gals, DecP_Gals, 10) 

		if Num is None: Num = self.N90
		tileData = self.tileData[:Num]

		Dec_tile = (tileData['dec_center']).values
		RA_tile = (tileData['ra_center']).values
		ID = (tileData['ID']).values

		for ii in range(len(tileData)):
			RAP_peak, DecP_peak = m(RA_tile[ii], Dec_tile[ii])

			m.plot(RAP_peak, DecP_peak, 'ro', markersize=3, mew=1, alpha=1)
			if tileEdge:
				RAP1, DecP1 = m((tileData['ra_Left_U'].values)[ii], (tileData['Dec_Up'].values)[ii])
				RAP2, DecP2 = m((tileData['ra_Right_U'].values)[ii], (tileData['Dec_Up'].values)[ii])
				RAP3, DecP3 = m((tileData['ra_Left_D'].values)[ii], (tileData['Dec_Down'].values)[ii])
				RAP4, DecP4 = m((tileData['ra_Right_D'].values)[ii], (tileData['Dec_Down'].values)[ii])

				m.plot([RAP1, RAP2], [DecP1, DecP2],'r-', linewidth=2, alpha=1) 
				m.plot([RAP2, RAP4], [DecP2, DecP4],'r-', linewidth=2, alpha=1) 
				m.plot([RAP4, RAP3], [DecP4, DecP3],'r-', linewidth=2, alpha=1) 
				m.plot([RAP3, RAP1], [DecP3, DecP1],'r-', linewidth=2, alpha=1)
		pl.show()
		return 0






