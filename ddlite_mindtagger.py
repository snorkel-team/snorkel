import atexit, json, os, pipes, socket, sys, time

class MindTaggerInstance:

  def __init__(self, mindtagger_format):
    self._mindtagger_format = mindtagger_format
    self.instance = None
    atexit.register(self._kill_mindtagger)
  
  def _system(self, script):
      return os.system("bash -c %s" % pipes.quote(script))
    
  def _kill_mindtagger(self):
    if self.instance is not None:
      self._system("kill -TERM $(cat mindtagger.pid)")
    
  def _generate_task_format(self):
    return  """
            <mindtagger mode="precision">

              <template for="each-item">
                %(title_block)s
                with probability <strong>{{item.probability | number:3}}</strong>
                appeared in sentence {{item.sent_id}} of document {{item.doc_id}}:
                <blockquote>
                    <big mindtagger-word-array="item.words" array-format="json">
                        %(style_block)s
                    </big>
                </blockquote>

                <div>
                  <div mindtagger-item-details></div>
                </div>
              </template>

              <template for="tags">
                <span mindtagger-adhoc-tags></span>
                <span mindtagger-note-tags></span>
              </template>

            </mindtagger>
            """ % self._mindtagger_format
    

  def launch_mindtagger(self, task_name, generate_items, task_root="mindtagger",
                        task_recreate = True, **kwargs):
                            
    args = dict(
        task = task_name,
        task_root = task_root,
        # figure out hostname and task name from IPython notebook
        host = socket.gethostname(),
        # determine a port number based on user name
        port = hash(os.getlogin()) % 1000 + 8000,
      )
    args['task_path'] = "%s/%s" % (args['task_root'], args['task'])
    args['mindtagger_baseurl'] = "http://%(host)s:%(port)s/" % args
    args['mindtagger_url'] = "%(mindtagger_baseurl)s#/mindtagger/%(task)s" % args
    # quoted values for shell
    shargs = { k: pipes.quote(str(v)) for k,v in args.iteritems() }

    

    # install mindbender included in DeepDive's release
    print "Making sure MindTagger is installed. Hang on!"
    if self._system("""
      export PREFIX="$PWD"/deepdive
      [[ -x "$PREFIX"/bin/mindbender ]] ||
      bash <(curl -fsSL git.io/getdeepdive || wget -qO- git.io/getdeepdive) deepdive_from_release
    """ % shargs) != 0: raise OSError("Mindtagger could not be installed")

    if task_recreate or not os.path.exists(args['task_path']):
      # prepare a mindtagger task from the data this object is holding
      try:
        if self._system("""
          set -eu
          t=%(task_path)s
          mkdir -p "$t"
          """ % shargs) != 0: raise OSError("Mindtagger task could not be created")
        with open("%(task_path)s/mindtagger.conf" % args, "w") as f:
          f.write("""
            title: %(task)s Labeling task for estimating precision
            items: {
                file: items.csv
                key_columns: [ext_id]
            }
            template: template.html
            """ % args)
        with open("%(task_path)s/template.html" % args, "w") as f:
          f.write(self._generate_task_format())
        with open("%(task_path)s/items.csv" % args, "w") as f:
          # prepare data to label
          import csv
          items = generate_items()
          item = next(items)
          o = csv.DictWriter(f, fieldnames=item.keys())
          o.writeheader()
          o.writerow(item)
          for item in items:
            o.writerow(item)
      except:
        raise OSError("Mindtagger task data could not be prepared: %s" % str(sys.exc_info()))

    # launch mindtagger
    if self._system("""
      # restart any running mindtagger instance
      ! [ -s mindtagger.pid ] || kill -TERM $(cat mindtagger.pid) || true
      PORT=%(port)s deepdive/bin/mindbender tagger %(task_root)s/*/mindtagger.conf &
      echo $! >mindtagger.pid
      """ % shargs) != 0: raise OSError("Mindtagger could not be started")
    while self._system("curl --silent --max-time 1 --head %(mindtagger_url)s >/dev/null" % shargs) != 0:
      time.sleep(0.1)        

    self.instance = args
    return args['mindtagger_url']
  
  def open_mindtagger(self, generate_mt_items, num_sample = 100, **kwargs):

    def generate_items():
      return generate_mt_items(num_sample)      

    # determine a task name using hash of the items
    # See: http://stackoverflow.com/a/7823051/390044 for non-negative hexadecimal
    def tohex(val, nbits):
      return "%x" % ((val + (1 << nbits)) % (1 << nbits))
    task_name = tohex(hash(json.dumps([i for i in generate_items()])), 64)

    # launch Mindtagger
    url = self.launch_mindtagger(task_name, generate_items, **kwargs)

    # return an iframe
    from IPython.display import IFrame
    return IFrame(url, **kwargs)
    
  def get_mindtagger_tags(self):
    tags_url = "%(mindtagger_baseurl)sapi/mindtagger/%(task)s/tags.json?attrs=ext_id" % self.instance

    import urllib, json
    opener = urllib.FancyURLopener({})
    t = opener.open(tags_url)
    tags = json.loads(t.read())

    return tags
