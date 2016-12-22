from .utils import ProgressBar
from multiprocessing import Process, Queue, JoinableQueue
from Queue import Empty
import sys
from .models.meta import new_sessionmaker


Q_GET_TIMEOUT = 5

class UDF(Process):
    def __init__(self, x_queue=None, y_queue=None):
        """
        x_queue: A Queue of input objects to process; primarily for running in parallel
        y_queue: A Queue to collect output objects to be added to a many-to-many set
            * NOTE: This latter queue will no longer be needed when we get rid of many-to-many sets in v0.5...
        """
        Process.__init__(self)
        self.daemon  = True
        self.x_queue = x_queue
        self.y_queue = y_queue

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

                    # TODO: Why does this cause it to hang??
                    if self.y_queue is not None:
                        self.y_queue.put(y)

                # Mark the object as processed
                self.x_queue.task_done()
            except Empty:
                break
        self.session.commit()
        self.session.close()
    
    def apply(self, x):
        """This function takes in an object, and returns a generator / set / list"""
        raise NotImplementedError()


class UDFRunnerMP(object):
    """Class to run UDFs in parallel using simple queue-based multiprocessing setup"""
    def __init__(self, udf_class, **udf_kwargs):
        self.udf_class  = udf_class
        self.udf_kwargs = udf_kwargs
        self.udfs       = []

    def run(self, xs, parallelism=1, y_set=None):

        # If y_set is provided, we need to collect the y objects and add them to the set on this thread...
        # Note: another reason to get rid of many-to-many sets asap!
        y_queue = Queue() if y_set is not None else None

        # Fill a JoinableQueue with input objects
        # TODO: For low-memory scenarios, we'd want to limit total queue size here
        x_queue = JoinableQueue()
        for x in xs:
            x_queue.put(x)

        # Start UDF Processes
        for i in range(parallelism):
            udf = self.udf_class(x_queue=x_queue, y_queue=y_queue, **self.udf_kwargs)
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

        # Collect the output objects and add to the set
        if y_set is not None:
            while True:
                try:
                    y_set.append(y_queue.get(False))
                except Empty:
                    break
            session.commit()
            return y_set


class UDFRunner(object):
    """Class to run a single UDF single-threaded"""
    def __init__(self, udf_class):
        self.udf = udf_class()

    def run(self, xs, y_set=None, max_n=None):
        
        # Set up ProgressBar if possible
        if hasattr(xs, '__len__') or max_n is not None:
            N  = len(xs) if hasattr(xs, '__len__') else max_n
            pb = ProgressBar(N)
        else:
            N = -1
        
        # Run single-thread
        for i, x in enumerate(xs):

            # If applicable, update progress bar
            if N > 0:
                pb.bar(i)
                if i == max_n:
                    break

            # Apply the UDF and add to either the set or the session
            for y in self.udf.apply(x):
                if y_set is not None:
                    add_to_collection(y_set, y)
                else:
                    self.udf.session.add(y)

        # Commit
        if self.udf.session is not None:
            self.udf.session.commit()

        # Close the progress bar if applicable
        if N > 0:
            pb.bar(N)
            pb.close()


def add_to_collection(c, x):
    """Adds, appends or puts x into c"""
    if hasattr(c, 'put'):
        c.put(x)
    elif hasattr(c, 'add'):
        c.add(x)
    elif hasattr(c, 'append'):
        c.append(x)
    else:
        raise AttributeError("No put/add/append attribute found.")

