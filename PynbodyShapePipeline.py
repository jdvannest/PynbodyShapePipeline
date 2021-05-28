'''
Author: Jordan Van Nest
This script creates (or modifies) a dictionary file for dwarfs in Rom25 or RomC. The dictionary
contains the shape history calculated with the pynbody method. This script should be run for
each timestep, and each time it's run, it appends the existing data with the values from the
current timestep. The script "script.py" generates a bash script to run this code for each 
timestep and in order so that the file is organized in an intuitive way (times and shape data
starting at z0 and going back timesteps in order).
'''

import pynbody,pickle,sys,pymp,warnings,argparse,os
import numpy as np

def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
#Number of bins to use in the shape calculation
parser.add_argument("-n","--number",required=True,type=int)
#Target simulation to analyze
parser.add_argument("-s","--simulation",choices=['RomC','Rom25'],required=True)
#Method of shape calculation (Shell is prefered)
parser.add_argument("-f","--filter",choices=['Shell','Sphere'],required=True)
#Target timestep of simulation to analyze
parser.add_argument("-t","--timestep",required=True,type=str)
#Overwrite the existing datafile instead of appending to it 
#(should only be done when starting over with z0 step)
parser.add_argument("-o","--overwrite",action='store_true')
#Print progress to terminal
parser.add_argument("-v","--verbose",action='store_true')
args = parser.parse_args()

#Load in the appropriate shape calculation fucntion
if args.filter=='Shell':
    from modules.Custom import halo_shape_stellar_shell as halo_shape_stellar
else:
    from modules.Custom import halo_shape_stellar_sphere as halo_shape_stellar

if args.simulation=='RomC':
    #Set variables for RomC
    #Path to simulation file
    simpath=f"/myhome2/users/munshi/Romulus/h1.cosmo50/h1.cosmo50PLK.1536gst1bwK1BH.{args.timestep}"
    #load in z0 IDs for dwarf halos
    with open('/myhome2/users/vannest/pfe_backup/nobackupp2/UDG/RomCHalos.txt') as f:
        halolist = f.readlines()
    halolist = [int(i) for i in halolist]
    #Load in timestep numbers
    with open('/myhome2/users/vannest/PynbodyShapePipeline/Timesteps.RomC.txt') as f:
        timesteps = f.readlines()
    timesteps = [t.rstrip('\n') for t in timesteps]
    #Arange with z0 as first entry to follow the tangos .calculate_for_progenitors format
    timesteps.reverse()
    #Load in IDs of major progenitors at each timestep
    HaloIDs = pickle.load(open(f'/myhome2/users/vannest/PynbodyShapePipeline/HaloIDs.RomC.pickle','rb'))
    #Path to output data file
    fname = f'/myhome2/users/vannest/PynbodyShapeHistory.RomC.{args.filter}.nshells{args.number}.pickle'
else:
    #Set variables for Rom25
    simpath= f"/myhome2/users/munshi/Romulus/cosmo25/cosmo25p.768sg1bwK1BHe75.{args.timestep}"
    with open('/myhome2/users/vannest/pfe_backup/nobackupp2/UDG/Rom25Halos.txt') as f:
        halolist = f.readlines()
    halolist = [int(i) for i in halolist]
    with open('/myhome2/users/vannest/PynbodyShapePipeline/Timesteps.Rom25.txt') as f:
        timesteps = f.readlines()
    timesteps = [t.rstrip('\n') for t in timesteps]
    timesteps.reverse()
    HaloIDs = pickle.load(open(f'/myhome2/users/vannest/PynbodyShapePipeline/HaloIDs.Rom25.pickle','rb'))
    fname = f'/myhome2/users/vannest/PynbodyShapeHistory.Rom25.{args.filter}.nshells{args.number}.pickle'

#Optional: delete existing file to start over
if args.overwrite:
    os.system(f'rm {fname}')
#Load in existing Datafile, or create new one if it doesn't exist
try:
    Datafile = pickle.load(open(fname,'rb'))
    if args.verbose: print('Data File Loaded.')
except:
    if args.verbose: print('No Data File Found. Writing new one...')
    Datafile = {}
    for halo in halolist:
        Datafile[str(halo)] = {}
        Datafile[str(halo)]['b_star'] = []
        Datafile[str(halo)]['c_star'] = []
        Datafile[str(halo)]['t'] = []


if args.verbose: print(f'Writing timestep {args.timestep}: 0.00%')
#This is the shared memory array that stores all the calculations for this timestep
Data = pymp.shared.dict()
#Load in the simulation
s = pynbody.load(simpath)
s.physical_units()
h = s.halos(dosort=True)
#This array is essentially a shared scalar to track progress
prog=pymp.shared.array((1,),dtype=int)
#Set OMP_NUM_THREADS
numproc = 9
with pymp.Parallel(numproc) as pl:
    for i in pl.xrange(len(halolist)):
        #Load in the appropriate z0 Halo ID
        hnumz0 = halolist[i]
        #Find the index of this timestep in the list of all timesteps
        t_index = timesteps.index(args.timestep)
        #Use the index to find the ID of the halo's progenitor at the appropriate timestep
        hnum = HaloIDs[str(hnumz0)]['HaloID'][t_index]
        #This dictionary is a temporary housing for the halos data defined in the scope of this process
        #This "middle man" dict is needed since the pymp package doesn't work well with writing to sub
        #dictionaries, so the "current" dict will be put into the shared "Data" dict as a final step
        current = {}
        current['b_star'] = np.nan
        current['c_star'] = np.nan
        current['t'] = s.properties['time'].in_units('Gyr')
        current['done'] = True
        #If a halo has no progenitors at a timestep (typically the early ones), then its hnum will be -1
        #If the halo has a progenitor at this timestep, perform the analysis
        if hnum>0:
            current['done'] = False
            #Load in a copy of the halo (or its progenitor
            halo = h.load_copy(hnum)
            halo.physical_units()
            #Analysis is in a try loop so a failure doesn't abort the pipeline
            try:
                #Center the halo
                pynbody.analysis.angmom.faceon(halo)
                #Calculate the half-light radius
                Rhalf = pynbody.analysis.luminosity.half_light_r(halo,band='v')
                #Perform the Shape Calculation
                r,ba,ca,angle,Es,nstar,nstar_i = halo_shape_stellar(halo,N=args.number)
                #Find the bin closes to 2 half-light radii
                ind = np.where(np.abs(r-2*Rhalf)==min(np.abs(r-2*Rhalf)))[0][0]
                #Make sure this bin has a nonzero star count (or find the closest one that does)
                #Then write the halo's 2R_half data to the private dictionary
                loop = True
                while loop:
                    if nstar[ind]==0:
                        ind+=1
                    else:
                        current['b_star'] = ba[ind]
                        current['c_star'] = ca[ind]
                        loop = False
            except:
                #Simple statement so that something is in the except block
                err = 1
        if args.verbose: myprint(f'Writing timestep {args.timestep}: '
                                +f'{round(float(prog[0]+1)/len(halolist)*100,2)}%',clear=True)
        #Feed the private dictionary for this halo into the shared dictionary for the timestep
        with pl.lock:
            prog[0] += 1
            Data[str(hnumz0)] = current
            del current

#Append the data from this timestep to the Main Datafile
for halo in halolist:
    if not Data[str(halo)]['done']:
        Datafile[str(halo)]['t'].append(Data[str(halo)]['t'])
        Datafile[str(halo)]['b_star'].append(Data[str(halo)]['b_star'])
        Datafile[str(halo)]['c_star'].append(Data[str(halo)]['c_star'])

#Save the updated Datafile
out = open(fname,'wb')
pickle.dump(Datafile,out)
out.close()
if args.verbose: myprint(f'Timestep {args.timestep} Completed.',clear=True)