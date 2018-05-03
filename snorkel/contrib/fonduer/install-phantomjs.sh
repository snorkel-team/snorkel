system=$(python -c "from sys import platform; print platform")
rm -rf phantomjs
if [ $system == "darwin" ]; then
    phantomjs="phantomjs-2.1.1-macosx"
elif [ $system == "win32" ]; then
    phantomjs="phantomjs-2.1.1-windows"
elif [ $system == "linux2" ]; then
    linuxbits=$(uname -m)
    if [ $linuxbits == "i686" ]; then
        phantomjs="phantomjs-2.1.1-linux-i686"
    else
        phantomjs="phantomjs-2.1.1-linux-x86_64"
    fi
else
    echo "
    Your OS does not support phantomjs static build.
    To install phantomjs from source, please visit http://phantomjs.org/download.html"
    exit
fi

if [ $system == "linux2" ]; then
    url=https://bitbucket.org/ariya/phantomjs/downloads/$phantomjs.tar.bz2
else
    url=https://bitbucket.org/ariya/phantomjs/downloads/$phantomjs.zip
fi

if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi

if [ $system == "linux2" ]; then
    tar jxf $phantomjs.tar.bz2
    rm $phantomjs.tar.bz2
else
    unzip $phantomjs.zip
    rm $phantomjs.zip
fi

mv $phantomjs phantomjs

