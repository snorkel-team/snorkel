# Set & move to home directory
source set_env.sh
cd "$SNORKELHOME"

# Make sure the submodules are installed
git submodule update --init --recursive

# Make sure parser is installed
PARSER="parser/stanford-corenlp-3.6.0.jar"
if [ ! -f "$PARSER" ]; then
    read -p "CoreNLP [default] parser not found- install now? [y/n]   " yn
    case $yn in
        [Yy]* ) echo "Installing parser..."; ./install-parser.sh;;
        [Nn]* ) ;;
    esac
fi

# Launch jupyter notebook!
echo "Launching Jupyter Notebook..."
jupyter notebook --no-browser
