# Make sure current directory is on pythonpath
here=$(cd "$(dirname "$0")" && pwd)
echo "$here"
export PYTHONPATH="$PYTHONPATH:$here"

# Launch jupyter notebook!
jupyter notebook
