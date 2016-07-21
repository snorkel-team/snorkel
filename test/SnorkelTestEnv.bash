# Snorkel test environment setup script for .bats files

# define some useful derived names and paths
SnorkelSourceTree=$(dirname "$BATS_TEST_DIRNAME") # parent is the top of the source
SNORKELHOME=$SnorkelSourceTree
SnorkelTestBasePath=${BATS_TEST_FILENAME%.bats}
SnorkelTestBaseName=${SnorkelTestBasePath#$SnorkelSourceTree/}

setup() {
    # keep a log file of full stdout/err
    exec > >(tee >&1 "$BATS_TEST_FILENAME.test$BATS_TEST_NUMBER".log) 2>&1

    # load the environment
    . "$SNORKELHOME"/set_env.sh
}
