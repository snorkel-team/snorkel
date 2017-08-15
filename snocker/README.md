## Containerized snorkel
This fork of the snorkel repo has been modified slightly to be build and deployed in a Docker container. No clone of this repository is necessary, that will be performed in a the Docker image build. Just download the app in the [*snocker* directory](https://github.com/HazyResearch/snorkel/tree/master/snocker) or create a local dockerfile and copy the contents into that file.

Make sure Docker is installed on your run environment and that the Dockerfile is available on the run envinment in a directory named app. Then build the image.

```
$ docker build -f app/dockerfile -t snocker:0.6.2 .
```

Now you can run the app. Link the home directory to somewhere easy to find.

```
$ docker run -it --name snocker -p 8887:8887 -v ~/some/local/dir/mapped/to/snokel/projects:/home/snorkel/projects snocker:0.6.2
## hit esc key sequence: ctrl+p+q
```

You can now execute a bash command in the running container, move the snorkel directory, and run snorkel.

```
$ docker exec -it snocker bash
root@ab1234:/# cd /home/snorkel
root@ab1234:/snorkel# ./run.sh
```

Feel free to install CoreNLP if you plan to use that parser instead of spaCy. After the install runs, Jupyter Notebook will start and you will be prompted with a dialog asking you to copy and paste a url... something like this:

```
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://0.0.0.0:8887/?token=b40203286f7c49e021c0dc0767239129f6865ce83b93a559
```

**Copy just the token and direct your web browser to port 8887 on the server running Docker**. If running on your local machine, this will look something like this: `http://localhost:8887/`

You will be prompted to paste the token in as a password the first time you visit this Jupyter Notebook instance. After that the running container will remember you.

You can now key out of the running container with **ctrl+p+q** and get to work in the jupyter notebook. If the container havens to stop you can restart it. Just keep your working files in the projects directory and all your work will persist on the installed server if

That's pretty much it.

I find the [Docker cheatsheet](https://www.docker.com/sites/default/files/Docker_CheatSheet_08.09.2016_0.pdf) to be a pretty useful reference.
