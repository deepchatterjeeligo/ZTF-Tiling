# ZTF-Tiling
This repo contains the tiling codes specific to ZTF


Given a GW sky-map this code will generate the ranked tiles and then prioratize them based on their GW localization probabilities.

Here is a simple description of how it works:
skymapFile = 'bayestar.fits.gz' ### The GW sky-map file
tileObj = ztf_tiling.RankedTileGenerator(skymapFile) ### Initiating the tile object

In this example we are generating the priority list for one hour of observation. The constrain for the observation is such that
the highest time we will allow per tile is 1200 seconds, and the lowest time is 600 seconds. The amount of time spent depends on
the localization probability contained in this tile (See Equation 2 of arXiv:1708.06723)
df = tileObj.createTilePriorityList(T_dur=1*3600, T_int_up=1200, T_int_low=600)

Currently this does not take into consideration which tiles are observation from a given location.

The output is in this format:

	program_id	field_id	filter_id	ra	dec	exposure_time	subprogram_name
  
0	2	311	1	67.4	-16.70769	1200.0	TOO_EMGW

1	2	361	1	74.6	-16.70769	868.0	TOO_EMGW

2	2	360	1	59.0	-9.78462	600.0	TOO_EMGW

3	2	312	1	66.05882	-9.78462	600.0	TOO_EMGW

4	2	411	1	59.0	-2.86154	600.0	TOO_EMGW
