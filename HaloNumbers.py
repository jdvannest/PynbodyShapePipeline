import numpy as np
import pickle,argparse,os,sys
def myprint(string,clear=False):
    if clear:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K") 
    print(string)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--simulation",choices=['Rom25','RomC'],required=True)
args = parser.parse_args()

myprint('Loading in Simulation...')
if args.simulation == 'RomC':
    db = 'C'
    dbname = 'h1.cosmo50'
    timestep = 72
    with open('/myhome2/users/vannest/pfe_backup/nobackupp2/UDG/RomCHalos.txt') as f:
        halos = f.readlines()
    halos = [int(i) for i in halos]
else:
    db = '25'
    dbname = 'cosmo25'
    timestep = 124
    with open('/myhome2/users/vannest/pfe_backup/nobackupp2/UDG/Rom25Halos.txt') as f:
        halos = f.readlines()
    halos = [int(i) for i in halos]
os.environ['TANGOS_DB_CONNECTION'] = '/myhome2/users/munshi/Romulus/data_romulus'+db+'.working.db'
import tangos
rom = tangos.get_simulation(dbname)
myprint('Simulation Loaded.',clear=True)

Data = {}
prog = 0
print('Writing: 0.00%')
for hnum in halos:
    Data[str(hnum)] = {}
    hid,t = rom[-1][hnum].calculate_for_progenitors('halo_number()','t()')   
    hid_pad = np.pad(hid,(0,timestep-len(hid)),'constant',constant_values=-1) 
    t_pad = np.pad(t,(0,timestep-len(t)),'constant',constant_values=-1) 
    Data[str(hnum)]['HaloID'] = hid_pad
    Data[str(hnum)]['time'] = t_pad
    prog+=1
    myprint(f'Writing: {round(float(prog)/len(halos)*100,2)}%',clear=True)
out = open(f'/myhome2/users/vannest/Data/HaloIDs.{args.simulation}.pickle','wb')
pickle.dump(Data,out)
out.close()