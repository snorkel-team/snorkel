phantomjs="phantomjs-2.1.1-macosx"
url=https://bitbucket.org/ariya/phantomjs/downloads/$phantomjs.zip
wget $url 
unzip $phantomjs.zip
rm $phantomjs.zip
mv $phantomjs phantomjs