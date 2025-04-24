def get2MASSfiles(object,size=2.0,outdir='.',verbose=False):
    """
    Code to import files from 2MASS and store them as a tarball of compressed fits.gz
    Inputs:
      object - a string such as 'M31'
      size   - size of the box side in arcdegs
      outdir - directory where the final tarball is stored
    """
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from pyvo.dal import imagesearch
    import os

    # Search in the 2MASS archive
    pos = SkyCoord.from_name(object)
    table = imagesearch('https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?type=at&ds=asky&',
                        pos, size=size).to_table()
    table = table[(table['band'].astype('S') == 'K') & (table['format'].astype('S') == 'image/fits')]
    hdus =  [fits.open(url)[0] for url in table['download'].astype('S')]

    # Create list of images/headers

    from astropy.wcs import WCS
    list = []
    for i,hdu in enumerate(hdus):
        if verbose:
            print('.',end='')
        header = hdu.header
        array = hdu.data
        wcs = WCS(header)
        list.append((array,wcs))
        outfile = object+'_{0:03d}.fits.gz'.format(i)
        hdu.writeto(os.path.join(outdir,outfile))
    print('\nImported: ',len(list),' files')

    # Compress files and do a tar archive
    import tarfile
    from glob import glob as gb
    files = sorted(gb(os.path.join(outdir,object+'*.fits.gz')))
    with tarfile.open(os.path.join(outdir,object+'.tar'), "w:tar") as tar:
        for file in files:
            if verbose:
                print('.', end='')
            tar.add(file)
            os.remove(file)
    if verbose:
        print('\nFiles archived into ', os.path.join(outdir,object+'.tar'))


def tarfile2healpix(object, outdir='.', verbose=False):
    from wfiskymatch.tools import fits2healpix
    import tarfile, os, gzip
    """
    Healpix project all fits files in a tar archive
    """
    outdirh5 = os.path.join(outdir,object+'_h5')
    print('Files will be in ', outdirh5)
    os.makedirs(outdirh5, exist_ok=True) 
    n = 0
    with tarfile.open(os.path.join(outdir,object+'.tar')) as tar:
        for item in tar.getnames():
            if item.endswith('fits.gz'):
                if verbose:
                    print('.',end='')
                tar.extract(item, '.',filter='data', numeric_owner=True)
                fits2healpix(item, outdirh5)
                os.remove(item) 
                n += 1
    if verbose:
        print('\nReprojected {0:n} images'.format(n))


def makeMosaic(object, outdir, offsetfile=False, verbose=False, savefootprint=False):
    """
    mosaic with astropy.reproject
    if offsetfile is present, the offsets are applied
    """

    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd
    from astropy.io import fits
    import tarfile, os, gzip
    import h5py
    import os
    import numpy as np
    from astropy.wcs import WCS
    from glob import glob as gb

    if offsetfile is False:
        files = []
        with tarfile.open(os.path.join(outdir,object+'.tar')) as tar:
            for item in tar.getnames():
                if item.endswith('fits.gz'):
                    files.append(item)
        offsets = np.zeros(len(files))
    else:
        h5file = os.path.join(outdir,object+'_offsets.h5')
        with h5py.File(h5file, 'r') as hdf5_file:
            offsets = hdf5_file['offsets'][:]
            files = hdf5_file['files'][:]
        # Remove the first / which is not conserved in the tar file 
        files = [os.path.join(outdir,f.decode('UTF-8')+'.fits.gz')[1:] for f in files]

    if verbose:
        print('Reading files ...')
    elist=[]
    i=0
    infile = os.path.join(outdir,object+'.tar')
    with tarfile.open(infile) as tar:
        for file, offset in zip(files, offsets):
            if verbose:
                print('.',end='')
            tar.extract(file, '.',filter='data', numeric_owner=True)
            with fits.open(file) as hdu:
                header = hdu[0].header
                array = hdu[0].data + offset
                wcs = WCS(header)
                elist.append((array,wcs))
            os.remove(file) 

    if verbose:
        print('\nReprojecting ...')
    from reproject.mosaicking import find_optimal_celestial_wcs
    from reproject import reproject_interp

    # Extent of the coaddition
    wcs_out, shape_out = find_optimal_celestial_wcs(elist)
    # Mosaic
    if offsetfile is False:
        match_background=True
    else:
        match_background=False        
    earray, footprint = reproject_and_coadd(elist,
                                            wcs_out, shape_out=shape_out,
                                            reproject_function=reproject_interp,
                                            match_background=match_background,
                                            blank_pixel_value=np.nan)
    idx = footprint == 0
    earray[idx] = np.nan

    # Save the result
    if verbose:
        print('Saving the result')
    if offsetfile is False:
        outfile = os.path.join(outdir,object+'_astropy.fits')
    else:
        outfile = os.path.join(outdir,object+'_wfiskymatch.fits')
    fits.writeto(outfile, earray, wcs_out.to_header(), overwrite=True)
    if savefootprint:
        outfile =  os.path.join(outdir,object+'_footprint.fits')
        fits.writeto(outfile, footprint, wcs_out.to_header(), overwrite=True)


def runMontage(object, outdir):
    """
    Script to run the montage code
    """
    import numpy as np
    import tarfile
    import os
    from astropy.io import fits
    
    montagedir = os.path.join(outdir,object+'_montage')
    os.makedirs(montagedir, exist_ok=True)
    kimagesdir = os.path.join(montagedir,'Kimages')
    os.makedirs(kimagesdir, exist_ok=True)

    files=[]
    with tarfile.open(os.path.join(outdir,object+'.tar')) as tar:
        for item in tar.getnames():
            if item.endswith('fits.gz'):
                #print('.',end='')
                filepath, filebase = os.path.split(item)
                tar.extract(item, '.',filter='data', numeric_owner=True)
                filename = os.path.join(kimagesdir,filebase)
                os.rename(item, filename)
                files.append(filename)
                # Eliminate nan
                with fits.open(filename, mode='update') as hdu:
                    data = hdu[0].data
                    idx = np.isfinite(data)
                    data[~idx] = np.nanmedian(data)

    # Unzip fits files
    bashCommand = "gunzip "+kimagesdir+"/*.fits.gz"
    os.system(bashCommand)

    # Generate a file with commands and source it
    with open(os.path.join(montagedir,'commands.sh'), 'w') as shfile:
        shfile.write('#!/bin/bash\n')
        shfile.write('export PATH="/Users/dfadda/Python/Montage/bin:$PATH"\n')
        shfile.write('cd '+montagedir+'\n')
        shfile.write('mkdir corrdir\n')
        shfile.write('mkdir diffdir\n')
        shfile.write('mkdir Kprojdir\n')
        shfile.write('mImgtbl Kimages Kimages.tbl\n')
        shfile.write('mMakeHdr Kimages.tbl Ktemplate.hdr\n')
        shfile.write('mProjExec -p Kimages Kimages.tbl Ktemplate.hdr Kprojdir Kstats.tbl\n')
        shfile.write('mImgtbl Kprojdir/ images.tbl\n')
        shfile.write('mAdd -p Kprojdir/ images.tbl Ktemplate.hdr '+object+'_uncorrected.fits\n')
        shfile.write('mOverlaps images.tbl diffs.tbl\n')
        shfile.write('mDiffExec -p Kprojdir/ diffs.tbl Ktemplate.hdr diffdir\n')
        shfile.write('mFitExec diffs.tbl fits.tbl diffdir\n')
        shfile.write('mBgModel images.tbl fits.tbl corrections.tbl\n')
        shfile.write('mBgExec -p Kprojdir/ images.tbl corrections.tbl corrdir\n')
        shfile.write('mAdd -p corrdir/ images.tbl Ktemplate.hdr '+object+'_montage.fits\n') 
        shfile.write('mv '+object+'_montage.fits ..\n')
        shfile.write('cd ..\n')
        shfile.write('\\rm -fr '+montagedir+'\n')

    bashCommand = "source "+os.path.join(montagedir,"commands.sh")
    os.system(bashCommand)

def runSkymatch(object, outdir):

    import numpy as np
    import tarfile
    import os
    from astropy.io import fits
    
    skymatchdir = os.path.join(outdir,object+'_skymatch')
    os.makedirs(skymatchdir, exist_ok=True)

    files=[]
    with tarfile.open(os.path.join(outdir,object+'.tar')) as tar:
        for item in tar.getnames():
            if item.endswith('fits.gz'):
                #print('.',end='')
                filepath, filebase = os.path.split(item)
                tar.extract(item, '.',filter='data', numeric_owner=True)
                filename = os.path.join(skymatchdir,filebase)
                os.rename(item, filename)
                files.append(filename)
                # Eliminate nan
                with fits.open(filename, mode='update') as hdu:
                    data = hdu[0].data
                    idx = np.isfinite(data)
                    data[~idx] = np.nanmedian(data)
    
    # Unzip fits files
    bashCommand = "gunzip "+skymatchdir+"/*.fits.gz"
    os.system(bashCommand)
    
    from stsci.skypac import skymatch
    from glob import glob as gb
    from astropy.wcs import WCS

    # Create list:
    files = sorted(gb(os.path.join(skymatchdir,'*.fits')))
    list = '[0],'.join(files)
    list += '[0]'
    # print(list)
    skymatch.skymatch(list, readonly=False, subtractsky=True, verbose=False)
    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd
    
    elist=[]
    for file in files:
        with fits.open(file) as hdu:
            header = hdu[0].header
            array = hdu[0].data
            wcs = WCS(header)
            elist.append((array,wcs))
    
    from reproject.mosaicking import find_optimal_celestial_wcs
    from reproject import reproject_interp
    
    # Extent of the coaddition
    wcs_out, shape_out = find_optimal_celestial_wcs(elist)
    
    #print('projection with offsets')
    earray, footprint = reproject_and_coadd(elist,
                                            wcs_out, shape_out=shape_out,
                                            reproject_function=reproject_interp,
                                            blank_pixel_value=np.nan)
    # Save the result
    idx = footprint == 0
    earray[idx] = np.nan
    fits.writeto(os.path.join(outdir,object+'_skymatch.fits'), earray, wcs_out.to_header(), overwrite=True)

    # Remove temporary skymatch directory
    bashCommand = "\\rm -fr "+skymatchdir
    os.system(bashCommand)

def plotcoverage(files, labels=None, cmap=None):
    """
    Given a list of hdf5 files plots the coverage over the sky in Mollweide projection
    """
    import numpy as np
    import h5py
    from healpy.newvisufunc import projview, newprojplot
    NPIX = 12*(2**10)**2
    coverage = np.arange(NPIX) * 0




    
    if labels is None:
        for file in files:
            with h5py.File(file, 'r') as hdf5_file:
                cov = hdf5_file['hpcoverage'][:]
                coverage[cov] = 1
    else:
        if len(files) == len(labels):
            for file,label in zip(files, labels):
                with h5py.File(file, 'r') as hdf5_file:
                    cov = hdf5_file['hpcoverage'][:]
                    coverage[cov] = label+1
        else:
            print('Labels do not correspond to files')
            
    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['white', 'orangered', 'chocolate', 'darkorange', 'orange', 'gold',
                  'yellowgreen','forestgreen','cyan','blue','violet','purple','black']
        cmap_name = 'my_list'
        maxcov = np.nanmax(coverage)+1
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors[:maxcov], N=maxcov)


    projview(
        coverage, coord=["C"], graticule=True, graticule_labels=True, projection_type="mollweide", nest=True,cmap=cmap
    )
    
