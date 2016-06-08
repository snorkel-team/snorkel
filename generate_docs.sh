# Refresh documentation for all python files in root dir
export PYTHONPATH="${PYTHONPATH}:."
for f in *.py; do
    echo -e "\n$f"
    pdoc --overwrite --html --html-dir docs/ $f
done
