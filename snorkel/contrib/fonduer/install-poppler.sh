# Build from source
POPPLER="poppler-0.53.0"
rm -rf poppler
url=https://poppler.freedesktop.org/${POPPLER}.tar.xz
if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi
tar -xf ${POPPLER}.tar.xz
rm ${POPPLER}.tar.xz
mv $POPPLER poppler
cd poppler
./configure --prefix="$PWD"
make
make install
cd ..
