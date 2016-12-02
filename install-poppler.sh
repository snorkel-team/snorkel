# Build from source
POPPLER="poppler-0.44.0"
rm -rf poppler
url=https://poppler.freedesktop.org/${POPPLER}.tar.xz
if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi
tar -xf ${POPPLER}.tar.xz
rm ${POPPLER}.tar.xz
cd $POPPLER
./configure --enable-poppler-glib
make
sudo make install
cd ..
mv $POPPLER poppler


#system=$(python -c "from sys import platform; print platform")
#if [ "$system" == "darwin" ]; then
#    brew update
#    brew install poppler
#else
#    sudo apt-get update
#    sudo apt-get install poppler-utils
#fi