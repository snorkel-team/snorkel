# Set & move to Snorkel home directory
source set_env.sh
cd "$SNORKELHOME"

# Make sure the submodules are installed
git submodule update --init --recursive

# Make sure parser is installed
PARSER="parser/stanford-corenlp-3.6.0.jar"
if [ ! -f "$PARSER" ]; then
    read -p "CoreNLP [default] parser not found- install now?  [y/n] " yn
    case $yn in
        [Yy]* ) echo "Installing parser..."; ./install-parser.sh;;
        [Nn]* ) ;;
    esac
fi

# Move to Fonduer home directory
cd "$FONDUERHOME"

# Make sure phantomjs is installed
PHANTOMJS="phantomjs/bin/phantomjs"
if [ ! -f "$PHANTOMJS" ]; then
    read -p "phantomjs not found- install now?  [y/n] " yn
    case $yn in
        [Yy]* ) echo "Installing phantomjs..."; ./install-phantomjs.sh;;
        [Nn]* ) ;;
    esac
fi

# Make sure poppler is installed
POPPLER="poppler/bin/pdftotext"
if [ ! -f "$POPPLER" ]; then
    read -p "poppler not found- install now?  [y/n] " yn
    case $yn in
        [Yy]* ) echo "Installing poppler..."; ./install-poppler.sh;;
        [Nn]* ) ;;
    esac
fi

# Move to snorkel directory
cd "$FONDUERHOME"

# Launch jupyter notebook!
echo "Launching Jupyter Notebook..."
jupyter notebook --notebook-dir="$FONDUERHOME"
