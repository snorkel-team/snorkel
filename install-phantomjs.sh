system=$(python -c "from sys import platform; print platform")
rm -rf phantomjs
if [ "$system" == "darwin" ]; then
    phantomjs="phantomjs-2.1.1-macosx"
elif [ $system == "win32" ]; then
    phantomjs="phantomjs-2.1.1-windows.zip"
elif [ $system == "linux2" ]; then
    linuxbits=$(uname -m)
    if [ $linuxbits == "i686" ]; then
        phantomjs="phantomjs-2.1.1-linux-i686.tar.bz2"
    else
        phantomjs="phantomjs-2.1.1-linux-x86_64.tar.bz2"
    fi
else
    echo "
    Your OS does not support phantomjs static build.
    To install phantomjs from source, please visit http://phantomjs.org/download.html"
    exit
fi

url=https://bitbucket.org/ariya/phantomjs/downloads/$phantomjs.zip
if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi
unzip $phantomjs.zip
rm $phantomjs.zip
mv $phantomjs phantomjs

