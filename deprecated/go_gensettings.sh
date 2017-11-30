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

# Run setup
# python -u snorkel/contrib/babble/pipelines/run.py \
#     --domain $DOMAIN \
#     --end_at 6 \
#     --debug \
#     --verbose --no_plots

# Run tests
for lf_propensity in True False 
do
for lf_prior in True False 
do
for lf_class_propensity in True False 
do
for class_prior in True False 
do
    echo ""
    echo "<TEST: Running with following params:>"
    echo "lf_propensity = $lf_propensity" 
    echo "lf_prior = $lf_prior"
    echo "lf_class_propensity = $lf_class_propensity"
    echo "class_prior = $class_prior"
    echo ""
    python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
        --reports_dir $REPORTS_DIR \
        --gen_init_params:lf_propensity $lf_propensity \
        --gen_init_params:lf_prior $lf_prior \
        --gen_init_params:lf_class_propensity $lf_class_propensity \
        --gen_init_params:class_prior $class_prior \
        --start_at 6 \
        --end_at 7 \
        --gen_model_search_space 2 \
        --seed 926 --verbose --no_plots |& tee -a $LOGFILE  # seed = 111
        # --parallelism 15 \
        # --postgres \
        # --debug \
done
done
done
done
