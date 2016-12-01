phantomjs="phantomjs-2.1.1-macosx"
url=https://bitbucket.org/ariya/phantomjs/downloads/$phantomjs.zip
if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi
unzip $phantomjs.zip
rm $phantomjs.zip
mv $phantomjs phantomjs