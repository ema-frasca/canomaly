# bin/bash
# download results from remote server
# run it in canomaly folder
# remote folder /nas/softechict-nas-1/rbenaglia/canomaly-data/results/dataset-can-mnist/classes_per_task-1/logs.pyd
# local folder ./storage/results/dataset-can-mnist/classes_per_task-1/logs.pyd

if [ -z "$1" ]
  then
    echo "No remote filename supplied"
    filename=/nas/softechict-nas-1/rbenaglia/canomaly-data/results/dataset-can-mnist/classes_per_task-1/logs.pyd
  else
    filename=$1
fi

if [ -z "$2" ]
  then
    echo "No local filename supplied"
    filename_local=logs.pyd
  else
    filename_local=$2
fi
sftp -r rbenaglia@aimagelab-srv-00.ing.unimore.it:"$filename" "$filename_local"
#if [[ -d $1 ]]; then
#
#else
#    sftp rbenaglia@aimagelab-srv-00.ing.unimore.it:"$filename" "$filename_local"
#fi
#if [ "${$ADDR: -1}" == *"."*]
#  then
#    sftp -r rbenaglia@aimagelab-srv-00.ing.unimore.it:"$filename" "$filename_local"
#    else
#      sftp rbenaglia@aimagelab-srv-00.ing.unimore.it:"$filename" "$filename_local"
#fi
