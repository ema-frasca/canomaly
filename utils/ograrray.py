import os
import subprocess
import sys
from argparse import ArgumentParser
from random import randint

local = False if 'SSH_CONNECTION' in os.environ else True

# launch this inside canomaly

parser = ArgumentParser(description='ograrray', allow_abbrev=False)
parser.add_argument('--path', type=str, required=True, help='list_path')
parser.add_argument('--nickname', type=str, default=None, help='Job name.')
parser.add_argument('--cycles', type=int, default=1,
                    help='The number of cycles.')
parser.add_argument('--max_jobs', type=int, default=None,
                    help='The maximum number of jobs.')
parser.add_argument('--skip_first', type=int, default=0,
                    help='Skips the first skip_first jobs.')
parser.add_argument('--exclude', type=str, default='', help='excNodeList.')
parser.add_argument('--n_gpus', type=int, default=1, help='Number of requested GPUs per job')
parser.add_argument('--custom_time', default=None, type=str, help='customizes sbatch time')
# parser.add_argument('--jobs_per_gpu', default=1, type=int, help='customizes sbatch time')
parser.add_argument('--user_name', type=str, default='rbenaglia')
parser.add_argument('--envname', type=str, default='env1')
parser.add_argument('--sbacciu', action='store_true')
# parser.add_argument('--rand_ord', action='store_true')
args = parser.parse_args()

red = int(args.cycles)

with open(args.path) as f:
    sbacci = f.read().splitlines()

if not local:
    interpreter = '/homes/%s/.conda/envs/%s/bin/python' % (args.user_name, args.envname)
else:
    interpreter = sys.executable

sbacci = [f'\'{interpreter} main.py ' + x + ' --logs \'' for x in sbacci if
          not x.startswith('#') and len(x.strip())] * red
sbacci = sbacci[args.skip_first:]

max_jobs = args.max_jobs if args.max_jobs is not None else len(sbacci)
nickname = args.nickname if args.nickname is not None else 'der-verse'

if not local:
    # if args.jobs_per_gpu == 1:
    gridsh = '''#!/bin/bash
    #SBATCH -p prod
    #SBATCH --job-name=<nick>
    #SBATCH --array=0-<lung>%<mj>
    #SBATCH --nodes=1
    <time>
    #SBATCH --output="/homes/<user_name>/output/<nick>_%A_%a.out"
    #SBATCH --error="/homes/<user_name>/output/<nick>_%A_%a.err"
    #SBATCH --gres=gpu:<ngpu>
    <xcld>
    
    arguments=(
    <REPLACEME>
    )
    
    sleep $(($RANDOM % 20)); ${arguments[$SLURM_ARRAY_TASK_ID]}
        '''
    # else:
    #     gridsh = '''#!/bin/bash
    # #SBATCH -p prod
    # #SBATCH --job-name=<nick>
    # #SBATCH --array=0-<lung>%<mj>
    # #SBATCH --nodes=1
    # <time>
    # #SBATCH --output="/homes/efrascaroli/output/<nick>_%A_%a.out"
    # #SBATCH --error="/homes/efrascaroli/output/<nick>_%A_%a.err"
    # #SBATCH --gres=gpu:<ngpu>
    # <xcld>
    #
    # arguments=(
    # <REPLACEME>
    # )
    #
    #
    # readarray -d : -t strarr <<< "${arguments[$SLURM_ARRAY_TASK_ID]}"
    # sleep $(($RANDOM % 20)); ${strarr[0]}& sleep 3; ${strarr[1]}
    #     '''

    gridsh = gridsh.replace('<REPLACEME>', '\n'.join(sbacci))
    gridsh = gridsh.replace('<lung>', str(len(sbacci) - 1))
    gridsh = gridsh.replace('<xcld>', ('#SBATCH --exclude=' + args.exclude) if len(args.exclude) else '')
    gridsh = gridsh.replace('<nick>', nickname)
    gridsh = gridsh.replace('<mj>', str(max_jobs))
    gridsh = gridsh.replace('<ngpu>', str(args.n_gpus))
    gridsh = gridsh.replace('<user_name>', args.user_name)
    gridsh = gridsh.replace('<time>', ('#SBATCH --time=' + args.custom_time) if args.custom_time is not None else '')

    with open('/homes/rbenaglia/sbatch/ready_for_sbatch.sh', 'w') as f:
        f.write(gridsh)
    print(gridsh)
    if args.sbacciu:
        os.system('sbatch /homes/rbenaglia/sbatch/ready_for_sbatch.sh')

else:
    if args.sbacciu:
        for c in sbacci:
            res = subprocess.run(
                c.replace("'", ''), check=True, shell=True
            )
