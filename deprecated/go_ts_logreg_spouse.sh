source set_env.sh

DOMAIN=$1
EXP=$2

DATE=`date +"%m_%d_%y"`
TIME=`date +"%H_%M_%S"`
LOGDIR="logs/$DATE"
mkdir -p $LOGDIR
LOGFILE="$LOGDIR/run_log_${DOMAIN}_${EXP}_${TIME}.log"
echo "Saving log to '$LOGFILE'"

REPORTS_DIR="reports/$DATE/${DOMAIN}_${EXP}/"
mkdir -p $REPORTS_DIR
echo "Saving reports to '$REPORTS_DIR'"
echo ""
echo "Note: If you are not starting at stage 0, confirm database exists already."

# Run tests
for MAX_TRAIN in 10 25 50 75 100 150 200 250 300 500 750 1000 1500 2000 3000 5000 7500 10000 12500 15000 17500 20000 22500
do
    echo ""
    echo "<TEST: Running with following params:>"
    echo "max_train = $MAX_TRAIN" 
    echo ""
    python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
        --reports_dir $REPORTS_DIR \
        --start_at 8 \
        --end_at 10 \
        --supervision traditional \
        --max_train $MAX_TRAIN \
	--gen_model_search_space 1 \
        --disc_model_search_space 1 \
	--disc_params_default:batch_size 64 \
	--disc_params_default:n_epochs 30 \
	--disc_params_default:lr 0.001 \
	--disc_params_default:rebalance 0.25 \
        --seed 123 --verbose --no_plots |& tee -a $LOGFILE
done
