from .utils import ProgressBar
from multiprocessing import Process, JoinableQueue
from Queue import Empty
from .models.meta import new_sessionmaker


QUEUE_TIMEOUT = 3


class UDFRunner(object):
    """Class to run UDFs in parallel using simple queue-based multiprocessing setup"""
    def __init__(self, udf_class, **udf_init_kwargs):
        self.udf_class       = udf_class
        self.udf_init_kwargs = udf_init_kwargs
        self.udfs            = []
        self.udf0            = self.udf_class(**self.udf_init_kwargs) if hasattr(self.udf_class, 'reduce') else None

    def apply(self, xs, clear=True, parallelism=None, progress_bar=True, **kwargs):

        # Clear everything downstream of this UDF if requested
        kwargs['clear'] = clear
        if clear:
            SnorkelSession = new_sessionmaker()
            session = SnorkelSession()
            self.clear(session, **kwargs)
            session.commit()
            session.close()

        # Execute the UDF
        if parallelism is None or parallelism < 2:
            self.apply_st(xs, progress_bar, **kwargs)
        else:
            self.apply_mt(xs, parallelism, **kwargs)

    def clear(self, session, **kwargs):
        raise NotImplementedError()

    def apply_st(self, xs, progress_bar, **kwargs):
        """Run the UDF single-threaded, optionally with progress bar"""
        udf = self.udf_class(**self.udf_init_kwargs)

        # Set up ProgressBar if possible
        pb = ProgressBar(len(xs)) if progress_bar and hasattr(xs, '__len__') else None
        
        # Run single-thread
        for i, x in enumerate(xs):
            if pb:
                pb.bar(i)

            # Apply UDF and add results to the session
            for y in udf.apply(x, **kwargs):
                
                # Uf UDF has a reduce step, this will take care of the insert; else add to session
                if hasattr(self.udf_class, 'reduce'):
                    udf.reduce(y, **kwargs)
                else:
                    udf.session.add(y)

        # Commit session and close progress bar if applicable
        udf.session.commit()
        if pb:
            pb.bar(len(xs))
            pb.close()

    def apply_mt(self, xs, parallelism, **kwargs):
        """Run the UDF multi-threaded using python multiprocessing"""
        # Fill a JoinableQueue with input objects
        # TODO: For low-memory scenarios, we'd want to limit total queue size here
        in_queue = JoinableQueue()
        for x in xs:
            in_queue.put(x)

        # If the UDF has a reduce step, we collect the output of apply in a Queue
        out_queue = None
        if hasattr(self.udf_class, 'reduce'):
            out_queue = JoinableQueue()

        # Start UDF Processes
        for i in range(parallelism):
            udf              = self.udf_class(in_queue=in_queue, out_queue=out_queue, **self.udf_init_kwargs)
            udf.apply_kwargs = kwargs
            self.udfs.append(udf)

        # Start the UDF processes, and then join on their completion
        for udf in self.udfs:
            udf.start()

        # If there is a reduce step, do now on this thread
        if hasattr(self.udf_class, 'reduce'):
            while any([udf.is_alive() for udf in self.udfs]):
                while True:
                    try:
                        y = out_queue.get(True, QUEUE_TIMEOUT)
                        self.udf0.reduce(y, **kwargs)
                        out_queue.task_done()
                    except Empty:
                        break
                self.udf0.session.commit()
            self.udf0.session.close()

        # Otherwise just join on the UDF.apply actions
        else:
            for i, udf in enumerate(self.udfs):
                udf.join()


class UDF(Process):
    def __init__(self, in_queue=None, out_queue=None):
        """
        in_queue: A Queue of input objects to process; primarily for running in parallel
        """
        Process.__init__(self)
        self.daemon       = True
        self.in_queue     = in_queue
        self.out_queue    = out_queue

        # Each UDF starts its own Engine
        # See http://docs.sqlalchemy.org/en/latest/core/pooling.html#using-connection-pools-with-multiprocessing
        SnorkelSession = new_sessionmaker()
        self.session   = SnorkelSession()

    def run(self):
        """
        This method is called when the UDF is run as a Process in a multiprocess setting
        The basic routine is: get from JoinableQueue, apply, put / add outputs, loop
        """
        while True:
            try:
                x = self.in_queue.get(True, QUEUE_TIMEOUT)
                for y in self.apply(x):

                    # If an out_queue is provided, add to that, else add to session
                    if self.out_queue is not None:
                        self.out_queue.put(y, True, QUEUE_TIMEOUT)
                    else:
                        self.session.add(y)
                self.in_queue.task_done()
            except Empty:
                break
        self.session.commit()
        self.session.close()

    def apply(self, x, **kwargs):
        """This function takes in an object, and returns a generator / set / list"""
        raise NotImplementedError()
