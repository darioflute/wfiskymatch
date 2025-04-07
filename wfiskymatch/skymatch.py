# Routine for computing cubes of interpolated images

def reprojectAll(
    input_data,
    output_projection,
    shape_out=None,
    reproject_function=None,
    output_arrays=None,
    output_footprints=None,
    **kwargs,
):
    """
    Given a set of input images, reproject these to a single cube.

    This currently only works with 2-d images with celestial WCS.

    Parameters
    ----------
    input_data : iterable
        One or more input datasets to reproject and co-add. This should be an
        iterable containing one entry for each dataset, where a single dataset
        is one of:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is an `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object
            * An `~astropy.nddata.NDData` object from which the ``.data`` and
              ``.wcs`` attributes will be used as the input data.

    output_projection : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a `~astropy.wcs.WCS`
        or a `~astropy.io.fits.Header` instance.
    shape_out : tuple, optional
        If ``output_projection`` is a `~astropy.wcs.WCS` instance, the
        shape of the output data should be specified separately.
    input_weights : iterable
        If specified, this should be an iterable with the same length as
        ``input_data``, where each item is one of:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * An `~numpy.ndarray` array

    hdu_in : int or str, optional
        If one or more items in ``input_data`` is a FITS file or an
        `~astropy.io.fits.HDUList` instance, specifies the HDU to use.
    hdu_weights : int or str, optional
        If one or more items in ``input_weights`` is a FITS file or an
        `~astropy.io.fits.HDUList` instance, specifies the HDU to use.
    reproject_function : callable
        The function to use for the reprojection.
    combine_function : { 'mean', 'sum', 'median', 'first', 'last', 'min', 'max' }
        The type of function to use for combining the values into the final
        image. For 'first' and 'last', respectively, the reprojected images are
        simply overlaid on top of each other. With respect to the order of the
        input images in ``input_data``, either the first or the last image to
        cover a region of overlap determines the output data for that region.
    match_background : bool
        Whether to match the backgrounds of the images.
    background_reference : `None` or `int`
        If `None`, the background matching will make it so that the average of
        the corrections for all images is zero. If an integer, this specifies
        the index of the image to use as a reference.
    output_array : array or None
        The final output array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with ``shape_out`` or derived from the output projection.
    output_footprint : array or None
        The final output footprint array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with ``shape_out`` or derived from the output projection.
    **kwargs
        Keyword arguments to be passed to the reprojection function.

    Returns
    -------
    array : `~numpy.ndarray`
        The co-added array.
    footprint : `~numpy.ndarray`
        Footprint of the co-added array. Values of 0 indicate no coverage or
        valid values in the input image, while values of 1 indicate valid
        values.
    """

    import numpy as np
    from astropy.wcs import WCS
    from astropy.wcs.wcsapi import SlicedLowLevelWCS
    from reproject.utils import parse_input_data, parse_input_weights, parse_output_projection


    # Validate inputs

    if reproject_function is None:
        raise ValueError(
            "reprojection function should be specified with the reproject_function argument"
        )

    # Parse the output projection to avoid having to do it for each

    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    if output_arrays is not None and output_array.shape != shape_out:
        raise ValueError(
            "If you specify an output array, it must have a shape matching "
            f"the output shape {shape_out}"
        )
    if output_footprints is not None and output_footprint.shape != shape_out:
        raise ValueError(
            "If you specify an output footprint array, it must have a shape matching "
            f"the output shape {shape_out}"
        )

    # Start off by reprojecting individual images to the final projection

    ndata = len(input_data)
    output_arrays = np.ones((ndata,shape_out[0],shape_out[1])) * np.nan
    output_footprints = np.zeros((ndata,shape_out[0],shape_out[1]))

    hdu_in = None
    for idata in range(ndata):
        # We need to pre-parse the data here since we need to figure out how to
        # optimize/minimize the size of each output tile (see below).
        array_in, wcs_in = parse_input_data(input_data[idata], hdu_in=hdu_in)
        ny, nx = array_in.shape
        n_per_edge = 11
        xs = np.linspace(-0.5, nx - 0.5, n_per_edge)
        ys = np.linspace(-0.5, ny - 0.5, n_per_edge)
        xs = np.concatenate((xs, np.full(n_per_edge, xs[-1]), xs, np.full(n_per_edge, xs[0])))
        ys = np.concatenate((np.full(n_per_edge, ys[0]), ys, np.full(n_per_edge, ys[-1]), ys))
        xc_out, yc_out = wcs_out.world_to_pixel(wcs_in.pixel_to_world(xs, ys))

        # Determine the cutout parameters

        # In some cases, images might not have valid coordinates in the corners,
        # such as all-sky images or full solar disk views. In this case we skip
        # this step and just use the full output WCS for reprojection.

        if np.any(np.isnan(xc_out)) or np.any(np.isnan(yc_out)):
            imin = 0
            imax = shape_out[1]
            jmin = 0
            jmax = shape_out[0]
        else:
            imin = max(0, int(np.floor(xc_out.min() + 0.5)))
            imax = min(shape_out[1], int(np.ceil(xc_out.max() + 0.5)))
            jmin = max(0, int(np.floor(yc_out.min() + 0.5)))
            jmax = min(shape_out[0], int(np.ceil(yc_out.max() + 0.5)))

        if imax < imin or jmax < jmin:
            continue

        if isinstance(wcs_out, WCS):
            wcs_out_indiv = wcs_out[jmin:jmax, imin:imax]
        else:
            wcs_out_indiv = SlicedLowLevelWCS(
                wcs_out.low_level_wcs, (slice(jmin, jmax), slice(imin, imax))
            )

        shape_out_indiv = (jmax - jmin, imax - imin)

        # TODO: optimize handling of weights by making reprojection functions
        # able to handle weights, and make the footprint become the combined
        # footprint + weight map

        array, footprint = reproject_function(
            (array_in, wcs_in),
            output_projection=wcs_out_indiv,
            shape_out=shape_out_indiv,
            hdu_in=hdu_in,
            **kwargs,
        )
        print(idata, end=' ')
        output_arrays[idata,jmin:jmax,imin:imax]=array
        output_footprints[idata,jmin:jmax,imin:imax]=footprint
    print('')

    return output_arrays, output_footprints

     
    

def computeOffsetsold(images, footprints, sigmas):
    """
    Input:

    images, cube of interpolated images
    footprints, cube of footprints
    sigmas, cube of uncertainties

    Output:

    offsets, array of offsets
    
    """
    import numpy as np
    import scipy as sp

    nfiles = len(images)
    A = np.zeros((nfiles, nfiles))
    I = np.zeros((nfiles, nfiles))
    B = np.zeros(nfiles)

    for i in range(nfiles-1):
        sigma2_i = sigmas[i]**2
        # Create subcubes
        idx = np.where(footprints[i]>0)
        imin,imax = np.min(idx[0]), np.max(idx[0])
        jmin,jmax = np.min(idx[1]), np.max(idx[1])
        ifootprints = footprints[:,imin:imax,jmin:jmax]
        iimages = images[:,imin:imax,jmin:jmax]
        sigma2 = sigmas[:,imin:imax, jmin:jmax]**2
        sigma2_i = sigma2[i]
        image_i = iimages[i]
        print(' [', i,']: ', end='')
        for j in range(i+1, nfiles):
            idx = np.where((ifootprints[i]  == 1) & (ifootprints[j]  == 1))
            if np.sum(idx) > 0:
                print(j, end=',')
                sigma2_j = sigma2[j]
                image_j = iimages[j]
                A[i,j] = - np.sum(1/(sigma2_i[idx] + sigma2_j[idx]))
                A[j,i] = A[i,j]
                I[i,j] = - np.sum((image_i[idx]-image_j[idx])/(sigma2_i[idx] + sigma2_j[idx]))
                I[j,i] = - I[i,j]
    #for i in range(nfiles-1):
    for i in range(nfiles):
        A[i,i] = -np.sum(A[i,:])
        B[i] = np.sum(I[i,:])

    # Add noise to avoid singular matrix error
    noise = 1e-15*np.random.rand(nfiles, nfiles)
    A += noise
    
    # Put last epsilon to zero - does this really work ?
    #A[nfiles-1,nfiles-1] = 1
    #B[nfiles-1] = 0
   
    # Solve the linear system
    #epsilon = np.linalg.solve(A, B)
    epsilon = sp.linalg.solve(A, B, assume_a='sym')
    # epsilon = sp.sparse.linalg.spsolve(sp.sparse.csc_array(A), B)  # works better fro sparse matrix

    # Implement the global shift minimizing the offsets
    delta = - np.nanmedian(epsilon)
    print('\n Common shift ', delta)
    epsilon += delta

    return epsilon


def computeOffsets(files, outfile="offsets"):
    """
    Input:

      files, list of files with images and WCS (can be compressed fits)

    Output:

      offsets, array of offsets
    
    """
    from scipy.sparse import csc_array
    from scipy.sparse.linalg import spsolve
    import h5py
    import numpy as np
    
    row = []
    col = []
    data = []
    B = np.zeros(nfiles)
    D = np.zeros(nfiles)

    for i in range(nfiles-1):
        print('.', end='')
        with h5py.File(files[i], 'r') as hdf5_file:
            hpimage = hdf5_file['hpimage'][:]
            apix = hpimage['pixel']
            aval = hpimage['value']
            aunc2 = hpimage['unc']**2
        for j in range(i+1, nfiles):
            if xcorr[i,j] == 1:
                with h5py.File(files[j], 'r') as hdf5_file:
                    hpimage = hdf5_file['hpimage'][:]
                    bpix = hpimage['pixel']
                    bval = hpimage['value']
                    bunc2 = hpimage['unc']**2
                    common_elements, aidx, bidx = np.intersect1d(apix, bpix, return_indices=True)
                if len(common_elements) > 0:
                    a_ij = - np.sum(1/(aunc2[aidx] + bunc2[bidx]))
                    I_ij = - np.sum((aval[aidx]-bval[bidx])/(aunc2[aidx] + bunc2[bidx]))
                    col.extend([i,j])
                    row.extend([j,i])
                    data.extend([a_ij, a_ij])
                    B[i] += I_ij
                    B[j] -= I_ij
                    D[i] -= a_ij
                    D[j] -= a_ij
    
    ii = np.arange(nfiles)
    row.extend(ii)
    col.extend(ii)
    data.extend(D)
    A = csc_array((data, (row, col)), shape=(nfiles,nfiles))

    epsilon = spsolve(A, B)
    delta = - np.nanmedian(epsilon2)
    epsilon += delta

    # Save pixels with finite values
    import h5py
    with h5py.File(os.path.join(outdir,outfile+'.h5'), 'w') as hdf5_file:
        d = hdf5_file.create_dataset('offsets', (len(epsilon),), dtype='float32') 
        d[:] = epsilon

    return epsilon
    


def computeUncertainties(images, footprints, shape_out):
    import numpy as np

    nfiles = len(images)
    sigmas = np.zeros((nfiles,shape_out[0],shape_out[1]))
    #noise = []
    #zerolev = []
    for sigma, image, footprint in zip(sigmas, images, footprints):
        print('.',end=' ')
        iidx = np.where(footprint>0)
        imin,imax = np.min(iidx[0]), np.max(iidx[0])
        jmin,jmax = np.min(iidx[1]), np.max(iidx[1])
        ifootprints = footprints[:,imin:imax,jmin:jmax]
        iimage = image[imin:imax,jmin:jmax]
        ifootprint = footprint[imin:imax,jmin:jmax]
        idx = np.where(ifootprint)
        data = iimage[idx]
        idx = np.isfinite(data)
        med = np.nanmedian(data[idx])
        mad = np.nanmedian(np.abs(data[idx]-med))
        for k in range(10):
            residuals = data - med
            idx = np.where(np.abs(residuals) < 3 * mad)
            med = np.nanmedian(data[idx])
            mad = np.nanmedian(np.abs(data[idx]-med))
        #noise.append(mad)
        #zerolev.append(med)
        sigma[iidx] = mad
        #sigma[iidx] = 1
 
    return sigmas


def computeSigma(values):
    import numpy as np
    # Compute sigma
    med = np.median(values)
    mad = np.median(np.abs(values-med))
    for k in range(10):
        residuals = values-med
        idr = np.where(np.abs(residuals) < 3 * mad)
        med = np.median(values[idr])
        mad = np.median(np.abs(values[idr]-med))
    return mad    

def fits2healpix(infile, outdir):
    """
    Reproject an image to a Healpix tessellation with 3" pixels

    input:
       infile, 'name of fits file'
       outdir, 'name of output directory'

    The code creates files with list of pixels, values, and uncertainty in hdf5 format
    """
    import os
    import numpy as np
    # 0. Extract filename [assumed to be the part before the first dot from the left]
    filepath, filebase = os.path.split(infile)
    filename = filebase.split('.')[0]
    # 1. Read the fits file
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(infile) as hdu:
        header = hdu[0].header
        data = hdu[0].data
    wcs = WCS(header)
    # grid of ra-dec 
    ny, nx = np.shape(data)
    x, y = np.arange(nx), np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    # 2. Healsparse map
    from healsparse import healSparseMap as hspm
    import healpy as hp
    nside_coverage = 2**10  # 3.4' x 3.4' tiles
    nside_sparse = 2**16    # 3.2" x 3.2" pixels
    hsp_map = hspm.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=np.float64)
    # list of Healpix pixels 
    px_num = hp.ang2pix(hsp_map._nside_sparse, ra, dec, nest=True, lonlat=True)
    upx = np.unique(px_num)
    # 3. Interpolation and update
    from scipy.ndimage import map_coordinates
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    coord_system_out = 'icrs'
    radec = hp.pix2ang(hsp_map._nside_sparse, upx, nest=True)
    lon_out = radec[1]*180/np.pi
    lat_out = 90-radec[0]*180/np.pi
    world_out = SkyCoord(lon_out*u.degree, lat_out*u.degree, frame=coord_system_out)
    # Look up pixels in input WCS
    xinds, yinds = wcs.world_to_pixel(world_out)
    coords = np.array([yinds, xinds])
    ## order of the spline interpolation: 1 is bilinear (does not create too much border effects)
    values = map_coordinates(data, coords, output=None, order = 1, cval=np.nan)
    # Update only possible with float64 !
    hsp_map.update_values_pix(upx, values.astype(np.float64))
    # Save pixels with finite values
    import h5py
    idx = np.isfinite(values)
    upx = upx[idx]
    values = values[idx]
    # Compute sigma
    mad = computeSigma(values)
    sigmas = np.full(len(values), mad)
    hpimage = np.rec.array([upx,values,sigmas],
                      formats='int64,float32,float32',
                      names='pixel, value, unc')
    idcoverage = np.where(hsp_map.coverage_map>0)[0]  # Select covered tiles
    with h5py.File(os.path.join(outdir,filename+'.h5'), 'w') as hdf5_file:
        hdf5_file.create_dataset('hpimage', data=hpimage, compression="gzip")
        d = hdf5_file.create_dataset('hpcoverage', (len(idcoverage),), dtype='int64') 
        d[:] = idcoverage


def asdf2healpix(infile, outdir):
    """
    Reproject an ASDF image to a Healpix tessellation with 3" pixels

    input:
       infile, 'name of asdf file'
       outdir, 'name of output directory'

    The code creates files with list of pixels, values, and uncertainty in hdf5 format
    """
    import os
    import numpy as np
    # 0. Extract filename [assumed to be the part before the first dot from the left]
    filepath, filebase = os.path.split(infile)
    filename = filebase.split('.')[0]
    # 1. Read the ASDF file
    import roman_datamodels as rd
    from astropy.wcs import WCS
    with rd.open(infile) as dm:
        data = dm.data.copy()
        err = dm.err.copy()
        header = dm.meta.wcs.to_fits()[0]
    wcs = WCS(header)
    # grid of ra-dec 
    ny, nx = np.shape(data)
    x, y = np.arange(nx), np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    # 2. Healsparse map
    from healsparse import healSparseMap as hspm
    import healpy as hp
    nside_coverage = 2**10  # 3.4' x 3.4' tiles
    nside_sparse = 2**16    # 3.2" x 3.2" pixels
    hsp_map = hspm.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=np.float64)
    # list of Healpix pixels 
    px_num = hp.ang2pix(hsp_map._nside_sparse, ra, dec, nest=True, lonlat=True)
    upx = np.unique(px_num)
    # 3. Interpolation and update
    from scipy.ndimage import map_coordinates
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    coord_system_out = 'icrs'
    radec = hp.pix2ang(hsp_map._nside_sparse, upx, nest=True)
    lon_out = radec[1]*180/np.pi
    lat_out = 90-radec[0]*180/np.pi
    world_out = SkyCoord(lon_out*u.degree, lat_out*u.degree, frame=coord_system_out)
    # Look up pixels in input WCS
    xinds, yinds = wcs.world_to_pixel(world_out)
    coords = np.array([yinds, xinds])
    ## order of the spline interpolation: 1 is bilinear (does not create too much border effects)
    values = map_coordinates(data, coords, output=None, order = 1, cval=np.nan)
    errvalues = map_coordinates(err, coords, output=None, order = 1, cval=np.nan)
    # Update only possible with float64 !
    hsp_map.update_values_pix(upx, values.astype(np.float64))
    # Save pixels with finite values
    import h5py
    idx = np.isfinite(values)
    upx = upx[idx]
    values = values[idx]
    sigmas = errvalues[idx]
    hpimage = np.rec.array([upx,values,sigmas],
                      formats='int64,float32,float32',
                      names='pixel, value, unc')
    idcoverage = np.where(hsp_map.coverage_map>0)[0]  # Select covered tiles
    print('filename is ', filename)
    print('saving in '+os.path.join(outdir,filename+'.h5'))
    with h5py.File(os.path.join(outdir,filename+'.h5'), 'w') as hdf5_file:
        hdf5_file.create_dataset('hpimage', data=hpimage, compression="gzip")
        d = hdf5_file.create_dataset('hpcoverage', (len(idcoverage),), dtype='int64') 
        d[:] = idcoverage

def plotcoverage(files):
    """
    Given a list of hdf5 files plots the coverage over the sky in Mollweide projection
    """
    import numpy as np
    from healpy.newvisufunc import projview, newprojplot
    NPIX = 12*(2**10)**2
    coverage = np.arange(NPIX) * 0

    for file in files:
        with h5py.File(file, 'r') as hdf5_file:
            cov = hdf5_file['hpcoverage'][:]
            coverage[cov] = 1
        
    projview(
        coverage, coord=["C"], graticule=True, graticule_labels=True, projection_type="mollweide", nest=True,cmap='gray_r'
    )

    
