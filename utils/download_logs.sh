# bin/bash
# $1 define position of log
# $2 define pattern
if [ -z "$1" ]
  then
    FILEPATH=/nas/softechict-nas-1/rbenaglia/canomaly-data/logs/dataset-can-mnist/classes_per_task-1/logs.pyd
  else
    FILEPATH=$1
fi

if [ -z "$2" ]
  then
    PATTERN='b5eaef65-ac47-4b61-9e10-765a4d30a7d7'
  else
    PATTERN=$2
fi
SCRIPT="cat $FILEPATH | grep -m 1 '$PATTERN' > single_log.pyd"
echo "$SCRIPT"
ssh -l rbenaglia aimagelab-srv-00.ing.unimore.it "${SCRIPT}"