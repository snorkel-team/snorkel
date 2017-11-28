source set_env.sh

DOMAIN=$1
EXP=$2

DATE=`date +"%m_%d_%y"`
TIME=`date +"%H_%M_%S"`
LOGDIR="logs/$DATE"
mkdir -p $LOGDIR

REPORTS_DIR="reports/$DATE/${DOMAIN}_${EXP}/"
mkdir -p $REPORTS_DIR
echo "Saving reports to '$REPORTS_DIR'"
echo ""
echo "Note: If you are not starting at stage 0, confirm database exists already."

# Run tests
echo ""
echo "<TEST: Running with following params:>"
echo ""
LOGFILE="$LOGDIR/run_log_${DOMAIN}_${EXP}_${TIME}.log"
echo "Saving log to '$LOGFILE'"
DBNAME=babble_${DOMAIN}_${EXP}_final
echo "Using dbname '$DBNAME'"
cp babble_${DOMAIN}_labeled_tocopy.db ${DBNAME}.db
python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
	--db_name $DBNAME \
        --reports_dir $REPORTS_DIR \
	--supervision generative \
	--learn_deps false \
        --gen_init_params:lf_propensity true \
        --gen_init_params:lf_class_propensity false \
        --gen_init_params:class_prior false \
        --disc_model_class lstm \
        --start_at 7 \
        --end_at 10 \
        --gen_model_search_space 20 \
	--disc_model_search_space 20 \
        --verbose --no_plots |& tee -a $LOGFILE &
