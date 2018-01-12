from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import *

from multiprocessing import Process, JoinableQueue
from queue import Empty

from .models.meta import new_sessionmaker, snorkel_conn_string
from .utils import ProgressBar


QUEUE_TIMEOUT = 3


class UDFRunner(object):
    """Class to run UDFs in parallel using simple queue-based multiprocessing setup"""
    def __init__(self, udf_class, **udf_init_kwargs):
        self.udf_class       = udf_class
        self.udf_init_kwargs = udf_init_kwargs
        self.udfs            = []

        if hasattr(self.udf_class, 'reduce'):
            self.reducer = self.udf_class(**self.udf_init_kwargs)
        else:
            self.reducer = None

    def apply(self, xs, clear=True, parallelism=None, progress_bar=True, count=None, **kwargs):
        """
        Apply the given UDF to the set of objects xs, either single or multi-threaded, 
        and optionally calling clear() first.
        """
        # Clear everything downstream of this UDF if requested
        if clear:
            print("Clearing existing...")
            SnorkelSession = new_sessionmaker()
            session = SnorkelSession()
            self.clear(session, **kwargs)
            session.commit()
            session.close()

        # Execute the UDF
        print("Running UDF...")
        if parallelism is None or parallelism < 2:
            self.apply_st(xs, progress_bar, clear=clear, count=count, **kwargs)
        else:
            self.apply_mt(xs, parallelism, clear=clear, **kwargs)

    def clear(self, session, **kwargs):
        raise NotImplementedError()

    def apply_st(self, xs, progress_bar, count, **kwargs):
        """Run the UDF single-threaded, optionally with progress bar"""
        udf = self.udf_class(**self.udf_init_kwargs)

        # Set up ProgressBar if possible
        pb = None
        if progress_bar and hasattr(xs, '__len__') or count is not None:
            n = count if count is not None else len(xs)
            pb = ProgressBar(n)
        
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
            pb.close()
        
    def apply_mt(self, xs, parallelism, **kwargs):
        """Run the UDF multi-threaded using python multiprocessing"""
        if snorkel_conn_string.startswith('sqlite'):
            raise ValueError('Multiprocessing with SQLite is not supported. Please use a different database backend,'
                             ' such as PostgreSQL.')

        # Fill a JoinableQueue with input objects
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
                        self.reducer.reduce(y, **kwargs)
                        out_queue.task_done()
                    except Empty:
                        break
                self.reducer.session.commit()
            self.reducer.session.close()

        # Otherwise just join on the UDF.apply actions
        else:
            for i, udf in enumerate(self.udfs):
                udf.join()

        # Terminate and flush the processes
        for udf in self.udfs:
            udf.terminate()
        self.udfs = []


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

        # We use a workaround to pass in the apply kwargs
        self.apply_kwargs = {}

    def run(self):
        """
        This method is called when the UDF is run as a Process in a multiprocess setting
        The basic routine is: get from JoinableQueue, apply, put / add outputs, loop
        """
        while True:
            try:
                x = self.in_queue.get(True, QUEUE_TIMEOUT)
                for y in self.apply(x, **self.apply_kwargs):

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
