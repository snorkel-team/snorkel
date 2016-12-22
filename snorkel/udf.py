from .utils import ProgressBar
from multiprocessing import Process, Queue, JoinableQueue
from Queue import Empty
import sys
from .models.meta import new_sessionmaker


Q_GET_TIMEOUT = 5

class UDF(Process):
    def __init__(self, x_queue=None):
        """
        x_queue: A Queue of input objects to process; primarily for running in parallel
        """
        Process.__init__(self)
        self.daemon  = True
        self.x_queue = x_queue

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
                x = self.x_queue.get(True, Q_GET_TIMEOUT)
                for y in self.apply(x):
                    self.session.add(y)
                self.x_queue.task_done()
            except Empty:
                break
        self.session.commit()
        self.session.close()
    
    def apply(self, x):
        """This function takes in an object, and returns a generator / set / list"""
        raise NotImplementedError()


class UDFRunner(object):
    """Class to run UDFs in parallel using simple queue-based multiprocessing setup"""
    def __init__(self, udf_class, **udf_kwargs):
        self.udf_class  = udf_class
        self.udf_kwargs = udf_kwargs
        self.udfs       = []

    def apply(self, xs, parallelism=None, progress_bar=True):
        if parallelism is None or parallelism < 2:
            self.apply_st(xs, progress_bar=progress_bar)
        else:
            self.apply_mt(xs, parallelism)

    def apply_st(self, xs, progress_bar):
        """Run the UDF single-threaded, optionally with progress bar"""
        udf = self.udf_class(**self.udf_kwargs)

        # Set up ProgressBar if possible
        pb = ProgressBar(len(xs)) if progress_bar and hasattr(xs, '__len__') else None
        
        # Run single-thread
        for i, x in enumerate(xs):
            if pb:
                pb.bar(i)

            # Apply UDF and add results to the session
            for y in udf.apply(x):
                udf.session.add(y)

        # Commit session and close progress bar if applicable
        udf.session.commit()
        if pb:
            pb.bar(len(xs))
            pb.close()

    def apply_mt(self, xs, parallelism):
        """Run the UDF multi-threaded using python multiprocessing"""
        # Fill a JoinableQueue with input objects
        # TODO: For low-memory scenarios, we'd want to limit total queue size here
        x_queue = JoinableQueue()
        for x in xs:
            x_queue.put(x)

        # Start UDF Processes
        for i in range(parallelism):
            udf = self.udf_class(x_queue=x_queue, **self.udf_kwargs)
            self.udfs.append(udf)

        # Start the UDF processes, and then join on their completion
        for udf in self.udfs:
            udf.start()

        # Join on the processes all finishing!
        nU = len(self.udfs)
        for i, udf in enumerate(self.udfs):
            udf.join()
            sys.stdout.write("\r%s / %s threads done." % (i+1, nU))
            sys.stdout.flush()
        print "\n"
        sys.stdout.flush()
