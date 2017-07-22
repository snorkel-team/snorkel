export SNORKELHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Snorkel home directory: $SNORKELHOME"
export PYTHONPATH="$PYTHONPATH:$SNORKELHOME:$SNORKELHOME/treedlib"
if [[ $SPARK_HOME ]]; then
  export PYTHONPATH="$PYTHONPATH:$SPARK_HOME/python"
fi
echo "Using PYTHONPATH=${PYTHONPATH}"
export PATH="$PATH:$SNORKELHOME:$SNORKELHOME/treedlib"
echo "Environment variables set!"
