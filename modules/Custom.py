from pynbody import filt, util, config, array, units, transformation
from pynbody.analysis import cosmology, _com, profile
import numpy as np
import math
import logging
logger = logging.getLogger('pynbody.analysis.halo')
def halo_shape_stellar_shell(sim, N=100, rin=None, rout=None, bins='equal'):
    """
    Returns radii in units of ``sim['pos']``, axis ratios b/a and c/a,
    the alignment angle of axis a in radians, and the rotation matrix
    for homeoidal shells over a range of N halo radii.

    **Keyword arguments:**

    *N* (100): The number of homeoidal shells to consider. Shells with
    few particles will take longer to fit.

    *rin* (None): The minimum radial bin in units of ``sim['pos']``.
    Note that this applies to axis a, so particles within this radius
    may still be included within homeoidal shells. By default this is
    taken as rout/1000.

    *rout* (None): The maximum radial bin in units of ``sim['pos']``.
    By default this is taken as the largest radial value in the halo
    particle distribution.

    *bins* (equal): The spacing scheme for the homeoidal shell bins.
    ``equal`` initialises radial bins with equal numbers of particles,
    with the exception of the final bin which will accomodate remainders.
    This number is not necessarily maintained during fitting.
    ``log`` and ``lin`` initialise bins with logarithmic and linear
    radial spacing.

    Halo must be in a centered frame.
    Caution is advised when assigning large number of bins and radial
    ranges with many particles, as the algorithm becomes very slow.
    """

    #-----------------------------FUNCTIONS-----------------------------
    # Define an ellipsoid shell with lengths a,b,c and orientation E:
    def Ellipsoid(r, a,b,c, E):
        x,y,z = np.dot(np.transpose(E),[r[:,0],r[:,1],r[:,2]])
        return (x/a)**2 + (y/b)**2 + (z/c)**2

    # Define moment of inertia tensor:
    MoI = lambda r,m: np.array([[np.sum(m*r[:,i]*r[:,j]) for j in range(3)]\
                               for i in range(3)])

    # Splits 'r' array into N groups containing equal numbers of particles.
    # An array is returned with the radial bins that contain these groups.
    sn = lambda r,N: np.append([r[i*int(len(r)/N):(1+i)*int(len(r)/N)][0]\
                               for i in range(N)],r[-1])

    # Retrieves alignment angle:
    almnt = lambda E: np.arccos(np.dot(np.dot(E,[1.,0.,0.]),[1.,0.,0.]))
    #-----------------------------FUNCTIONS-----------------------------

    if (rout == None): rout = sim.s['r'].max()
    if (rin == None): rin = rout/1E3

    posr = np.array(sim.s['r'])[np.where(sim.s['r'] < rout)]
    pos = np.array(sim.s['pos'])[np.where(sim.s['r'] < rout)]
    mass = np.array(sim.s['mass'])[np.where(sim.s['r'] < rout)]

    rx = [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
    ry = [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    rz = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]

    # Define bins:
    if (bins == 'equal'): # Each bin contains equal number of particles
        mid = sn(np.sort(posr[np.where((posr >= rin) & (posr <= rout))]),N*2)
        rbin = mid[1:N*2+1:2]
        mid = mid[0:N*2+1:2]

    elif (bins == 'log'): # Bins are logarithmically spaced
        mid = profile.Profile(sim.s, type='log', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = np.sqrt(mid[0:N]*mid[1:N+1])

    elif (bins == 'lin'): # Bins are linearly spaced
        mid = profile.Profile(sim.s, type='lin', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = 0.5*(mid[0:N]+mid[1:N+1])

    # Define b/a and c/a ratios and angle arrays:
    ba,ca,angle,nstar,nstar_i = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
    Es = [0]*N

    # Begin loop through radii:
    for i in range(0,N):

        # Initialise convergence criterion:
        tol = 0.1
        count = 0

        # Define initial spherical shell:
        a=b=c = rbin[i]
        E = np.identity(3)
        L1,L2 = rbin[i]-mid[i],mid[i+1]-rbin[i]

        # Begin iterative procedure to fit data to shell:
        while True:
            count += 1

            # Collect all particle positions and masses within shell:
            r = pos[np.where((posr < a+L2) & (posr > c-L1*c/a))]
            inner = Ellipsoid(r, a-L1,b-L1*b/a,c-L1*c/a, E)
            outer = Ellipsoid(r, a+L2,b+L2*b/a,c+L2*c/a, E)
            r = r[np.where((inner > 1.) & (outer < 1.))]
            m = mass[np.where((inner > 1.) & (outer < 1.))]
            num = len(r)
            if (count == 1): num_i = len(r)

            # End iterations if there is no data in range:
            if (len(r) == 0):
                ba[i],ca[i],angle[i],Es[i],nstar_i[i] = b/a,c/a,almnt(E),E,num_i
                logger.info('No data in range after %i iterations' %count)
                break

            # Calculate shape tensor & diagonalise:
            D = list(np.linalg.eig(MoI(r,m)/np.sum(m)))

            # Purge complex numbers:
            if isinstance(D[1][0,0],complex):
                D[0] = D[0].real ; D[1] = D[1].real
                logger.info('Complex numbers in D removed...')

            # Compute ratios a,b,c from moment of intertia principles:
            anew,bnew,cnew = np.sqrt(abs(D[0])*3.0)

            # The rotation matrix must be reoriented:
            E = D[1]
            if ((bnew > anew) & (anew >= cnew)): E = np.dot(E,rz)
            if ((cnew > anew) & (anew >= bnew)): E = np.dot(np.dot(E,ry),rx)
            if ((bnew > cnew) & (cnew >= anew)): E = np.dot(np.dot(E,rz),rx)
            if ((anew > cnew) & (cnew >= bnew)): E = np.dot(E,rx)
            if ((cnew > bnew) & (bnew >= anew)): E = np.dot(E,ry)
            cnew,bnew,anew = np.sort(np.sqrt(abs(D[0])*3.0))

            # Keep a as semi-major axis and distort b,c by b/a and c/a:
            div = rbin[i]/anew
            anew *= div
            bnew *= div
            cnew *= div

            # Convergence criterion:
            if (np.abs(b/a-bnew/anew) < tol) & (np.abs(c/a-cnew/anew) < tol):
                if (almnt(-E) < almnt(E)): E = -E
                ba[i],ca[i],angle[i],Es[i],nstar[i],nstar_i[i] = bnew/anew,cnew/anew,almnt(E),E,num,num_i
                break

            # Increase tolerance if convergence has stagnated:
            elif (count%10 == 0): tol *= 5.

            # Reset a,b,c for the next iteration:
            a,b,c = anew,bnew,cnew

    return [array.SimArray(rbin, sim.d['pos'].units), ba, ca, angle, Es, nstar, nstar_i]









def halo_shape_stellar_sphere(sim, N=100, rin=None, rout=None, bins='equal'):
    """
    Returns radii in units of ``sim['pos']``, axis ratios b/a and c/a,
    the alignment angle of axis a in radians, and the rotation matrix
    for homeoidal shells over a range of N halo radii.

    **Keyword arguments:**

    *N* (100): The number of homeoidal shells to consider. Shells with
    few particles will take longer to fit.

    *rin* (None): The minimum radial bin in units of ``sim['pos']``.
    Note that this applies to axis a, so particles within this radius
    may still be included within homeoidal shells. By default this is
    taken as rout/1000.

    *rout* (None): The maximum radial bin in units of ``sim['pos']``.
    By default this is taken as the largest radial value in the halo
    particle distribution.

    *bins* (equal): The spacing scheme for the homeoidal shell bins.
    ``equal`` initialises radial bins with equal numbers of particles,
    with the exception of the final bin which will accomodate remainders.
    This number is not necessarily maintained during fitting.
    ``log`` and ``lin`` initialise bins with logarithmic and linear
    radial spacing.

    Halo must be in a centered frame.
    Caution is advised when assigning large number of bins and radial
    ranges with many particles, as the algorithm becomes very slow.
    """

    #-----------------------------FUNCTIONS-----------------------------
    # Define an ellipsoid shell with lengths a,b,c and orientation E:
    def Ellipsoid(r, a,b,c, E):
        x,y,z = np.dot(np.transpose(E),[r[:,0],r[:,1],r[:,2]])
        return (x/a)**2 + (y/b)**2 + (z/c)**2

    # Define moment of inertia tensor:
    MoI = lambda r,m: np.array([[np.sum(m*r[:,i]*r[:,j]) for j in range(3)]\
                               for i in range(3)])

    # Splits 'r' array into N groups containing equal numbers of particles.
    # An array is returned with the radial bins that contain these groups.
    sn = lambda r,N: np.append([r[i*int(len(r)/N):(1+i)*int(len(r)/N)][0]\
                               for i in range(N)],r[-1])

    # Retrieves alignment angle:
    almnt = lambda E: np.arccos(np.dot(np.dot(E,[1.,0.,0.]),[1.,0.,0.]))
    #-----------------------------FUNCTIONS-----------------------------

    if (rout == None): rout = sim.s['r'].max()
    if (rin == None): rin = rout/1E3

    posr = np.array(sim.s['r'])[np.where(sim.s['r'] < rout)]
    pos = np.array(sim.s['pos'])[np.where(sim.s['r'] < rout)]
    mass = np.array(sim.s['mass'])[np.where(sim.s['r'] < rout)]

    rx = [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
    ry = [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    rz = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]

    # Define bins:
    if (bins == 'equal'): # Each bin contains equal number of particles
        mid = sn(np.sort(posr[np.where((posr >= rin) & (posr <= rout))]),N*2)
        rbin = mid[1:N*2+1:2]
        mid = mid[0:N*2+1:2]

    elif (bins == 'log'): # Bins are logarithmically spaced
        mid = profile.Profile(sim.s, type='log', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = np.sqrt(mid[0:N]*mid[1:N+1])

    elif (bins == 'lin'): # Bins are linearly spaced
        mid = profile.Profile(sim.s, type='lin', ndim=3, rmin=rin, rmax=rout, nbins=N+1)['rbins']
        rbin = 0.5*(mid[0:N]+mid[1:N+1])

    # Define b/a and c/a ratios and angle arrays:
    ba,ca,angle,nstar,nstar_i = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
    Es = [0]*N

    # Begin loop through radii:
    for i in range(0,N):

        # Initialise convergence criterion:
        tol = 0.1
        count = 0

        # Define initial spherical shell:
        a=b=c = rbin[i]
        E = np.identity(3)
        L1,L2 = rbin[i]-mid[i],mid[i+1]-rbin[i]

        # Begin iterative procedure to fit data to shell:
        while True:
            count += 1

            # Collect all particle positions and masses within sphere:
            r = pos[np.where(posr < a+L2)]
            #inner = Ellipsoid(r, a-L1,b-L1*b/a,c-L1*c/a, E)
            outer = Ellipsoid(r, a+L2,b+L2*b/a,c+L2*c/a, E)
            r = r[np.where(outer < 1.)]
            m = mass[np.where(outer < 1.)]
            num = len(r)
            if (count == 1): num_i = len(r)

            # End iterations if there is no data in range:
            if (len(r) == 0):
                ba[i],ca[i],angle[i],Es[i],nstar_i[i] = b/a,c/a,almnt(E),E,num_i
                logger.info('No data in range after %i iterations' %count)
                break

            # Calculate shape tensor & diagonalise:
            D = list(np.linalg.eig(MoI(r,m)/np.sum(m)))

            # Purge complex numbers:
            if isinstance(D[1][0,0],complex):
                D[0] = D[0].real ; D[1] = D[1].real
                logger.info('Complex numbers in D removed...')

            # Compute ratios a,b,c from moment of intertia principles:
            anew,bnew,cnew = np.sqrt(abs(D[0])*3.0)

            # The rotation matrix must be reoriented:
            E = D[1]
            if ((bnew > anew) & (anew >= cnew)): E = np.dot(E,rz)
            if ((cnew > anew) & (anew >= bnew)): E = np.dot(np.dot(E,ry),rx)
            if ((bnew > cnew) & (cnew >= anew)): E = np.dot(np.dot(E,rz),rx)
            if ((anew > cnew) & (cnew >= bnew)): E = np.dot(E,rx)
            if ((cnew > bnew) & (bnew >= anew)): E = np.dot(E,ry)
            cnew,bnew,anew = np.sort(np.sqrt(abs(D[0])*3.0))

            # Keep a as semi-major axis and distort b,c by b/a and c/a:
            div = rbin[i]/anew
            anew *= div
            bnew *= div
            cnew *= div

            # Convergence criterion:
            if (np.abs(b/a-bnew/anew) < tol) & (np.abs(c/a-cnew/anew) < tol):
                if (almnt(-E) < almnt(E)): E = -E
                ba[i],ca[i],angle[i],Es[i],nstar[i],nstar_i[i] = bnew/anew,cnew/anew,almnt(E),E,num,num_i
                break

            # Increase tolerance if convergence has stagnated:
            elif (count%10 == 0): tol *= 5.

            # Reset a,b,c for the next iteration:
            a,b,c = anew,bnew,cnew

    return [array.SimArray(rbin, sim.d['pos'].units), ba, ca, angle, Es, nstar, nstar_i]