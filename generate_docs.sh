# Refresh documentation for all python files in root dir
export PYTHONPATH="${PYTHONPATH}:."
rm -rf docs
mkdir docs
for f in *.py; do
    echo -e "\n$f"
    pdoc --overwrite --html --html-dir docs/ $f
done
echo ""
echo "Done: PDoc documentation in docs/."
