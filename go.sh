source set_env.sh

DOMAIN=$1
START=$2

DATE=`date +"%m_%d_%y"`
TIME=`date +"%H_%M_%S"`
LOGDIR="logs/$DATE"
mkdir -p $LOGDIR
LOGFILE="$LOGDIR/run_log_${DOMAIN}_${TIME}.log"
echo "Saving log to '$LOGFILE'"

python -u snorkel/contrib/babble/pipelines/run.py \
    --domain $DOMAIN \
    --verbose \
    --no_plots \
    --start_at $START | tee $LOGFILE