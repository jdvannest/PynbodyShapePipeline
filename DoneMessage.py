import time,argparse,datetime,os

parser = argparse.ArgumentParser()
parser.add_argument("-t","--time",type=str,choices=['start','stop'])
parser.add_argument("-a","--append",type=str)
parser.add_argument("-s","--simulation",choices=['RomC','Rom25'],required=True)
args = parser.parse_args()

filepath = '/nobackup/jvannest/UDG/PynbodyShapePipeline/Done.txt'
#filepath = '/Users/jdvannest/Pynbody/UDG/PynbodyShapePipeline/Done.txt'

current = time.asctime().split()
currenttime = current[3].split(':')
currenttime[0] = str(int(currenttime[0])+2)
correctedtime = ':'.join(currenttime)
donetime = current[0]+', '+current[1]+' '+current[2]+' at '+correctedtime

if args.time=='start':
    f = open(filepath,'w')
    f.writelines(f'Analysis of {args.simulation} started at {donetime}\n')
    f.writelines(f'start time : {time.time()}\n')
    f.close()
elif args.append is not None:
    with open(filepath) as f:
        L = f.readlines()
    end = time.time()
    start = float(L[-1].split(':')[1])
    if L[-1].split()[0]=='st': del L[-1]
    dt = datetime.timedelta(seconds = (end-start))
    L.append(f'Timestep {args.append} Finished. Run Time: {dt}\n')
    L.append(f'st : {time.time()}\n')
    f = open(filepath,'w')
    f.writelines(L)
    f.close()
elif args.time=='stop':
    with open(filepath) as f:
        L = f.readlines()
    start = float(L[1].split(':')[1])
    del L[1]
    end = time.time()
    dt = datetime.timedelta(seconds = (end-start))
    if L[-1].split()[0]=='st': del L[-1]
    f = open(filepath,'w')
    f.writelines(f'Analysis of {args.simulation} finished at {donetime}\n')
    f.writelines(f'Total Run Time: {str(dt)}\n\n')
    f.writelines(L)
    f.close()
    os.system('sendmail jdvannest@ou.edu < Done.txt')