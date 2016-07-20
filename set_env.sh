export SNORKELHOME=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$PYTHONPATH:$SNORKELHOME:$SNORKELHOME/treedlib"
export PATH="$PATH:$SNORKELHOME:$SNORKELHOME/treedlib"
echo "Environment variables set!"
