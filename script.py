sims = ['Rom25','RomC']
n = 25
filt = 'Shell'
v = '-v'

for sim in sims:
    f = open(f'Pipeline_{sim}','w')
    f.writelines('#!/bin/bash\n')
    f.writelines(f'python DoneMessage.py -t start -s {sim}\n')
    times = ['008192']
    first = True
    for time in times:
        if first:
            f.writelines(f'python PynbodyShapePipeline.py {v} -n {n} -s {sim} -f {filt} -t {time} -o\n')
            first = False
        else:
            f.writelines(f'python PynbodyShapePipeline.py {v} -n {n} -s {sim} -f {filt} -t {time}\n')
        f.writelines(f'python DoneMessage.py -s Rom25 -a {time}\n')
        #f.writelines(f'echo Step {time} Finished\n')
    f.writelines(f'python DoneMessage.py -t stop -s {sim}')
    f.close()