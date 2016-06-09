# Set & move to home directory
export DDLHOME=$(cd "$(dirname "$0")" && pwd)
cd "$DDLHOME"

# Make sure home directory is on pythonpath
export PYTHONPATH="$PYTHONPATH:$DDLHOME"

# Make sure parser is installed
PARSER="parser/stanford-corenlp-3.6.0.jar"
if [ ! -f "$PARSER" ]; then
    read -p "CoreNLP [default] parser not found- install now?   " yn
    case $yn in
        [Yy]* ) echo "Installing parser..."; ./install-parser.sh;;
        [Nn]* ) ;;
    esac
fi

# Launch jupyter notebook!
echo "Launching Jupyter Notebook..."
jupyter notebook
