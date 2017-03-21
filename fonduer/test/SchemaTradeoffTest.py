import numpy as np
import timeit
import matplotlib.pyplot as plt
from collections import defaultdict

rng = np.random.rand
candidate_count = 10

# Parameter list
feat_sizes = [1000, 3000, 10000, 30000, 100000]
iterations = [1, 3, 10, 30, 100]
inv_densities = [1000, 100, 30, 10, 1]

default_inv_density = 100
default_feat_size = 100

def new_immutable_vec(size):
    return tuple(rng(size))

# n by m random matrix constructors
class lil():
    
    def __init__(self, n, m, sparsity = default_inv_density):
        m/=sparsity
        self.data = [new_immutable_vec(m) for _ in xrange(n)]
        
    def add_col(self, size):
        for i in xrange(len(self.data)):
            # Updates each row with new cols
            self.data[i] += new_immutable_vec(size)
            
    def query(self):
        return self.data[0]
        
class coo():
    
    def __init__(self, n, m, sparsity = default_inv_density):
        m/=sparsity
        # Simulate (id, key, val) schema
        self.data = [(i, j, rng()) for i in xrange(n) for j in xrange(m)]
        self.row_count = n
        self.col_count = m
        
    def add_col(self, size):
        self.col_count += size
        for r in xrange(self.row_count):
            for new_key in xrange(size):
                self.data.append((r, new_key, rng()))
                
    def query(self):
        return tuple(self.data[:self.col_count])
                
class dense():
    
    def __init__(self, n, m, sparsity = default_inv_density):
        # Simulate (id, key, val) schema
        self.data = np.ones([n, m])
        if m < sparsity:
            raise ValueError('Sparsity %d too great for columns %d' % (sparsity, m))
        rng(n, m/sparsity)
        self.n = n
        
    def add_col(self, size):
        # Effectively copy the old data into a newly allocated matrix
        self.data = np.append(self.data, np.random.rand(self.n, size), axis=1)
        
    def query(self):
        return self.data[0]

def show_plot(xname, yname):
    axes = plt.gca()
    caption = ''#%dx%d, density=%s' % (candidate_count, feat_count, str(1.0/default_inv_density))
    axes.set_title(caption)
    axes.set_xlabel(xname.title())
    axes.set_ylabel(yname.title())
    plt.xscale('log', nonposy='clip')
    plt.yscale('log', nonposy='clip')
    plt.legend(loc='upper left')
    plt.show()
    
def run_benchmark(xaxis, yaxis):    
    series = defaultdict(list)
    mat_types = [dense, coo, lil]
    real_xs = None
    if xaxis == 'feat_size':
        xs = feat_sizes
    elif xaxis == 'iterations':
        xs = iterations
    elif xaxis == 'sparsity':
        xs = inv_densities
        real_xs = [1.0/x for x in xs]
    if not real_xs: real_xs = xs

    repetition = 10
    exeuctions = 1        
    for mat in mat_types:
        mat_name = mat.__name__
        # x is the variable on x-axis for plotting
        for x in xs:
            m = None
            if xaxis == 'feat_size':
                if yaxis == 'mat_time':
                    run = lambda: mat(candidate_count, x)
                else:
                    m = mat(candidate_count, x)
            elif xaxis == 'iterations':
                if yaxis == 'mat_time':
                    def run():
                        # Here we simulate higher dimension with dense matrix
                        m = mat(candidate_count, 1000, 10)
                        for _ in xrange(x):
                            m.add_col(1)
                else:
                    # Here we directly instantiate the larger matrix for query
                    m = mat(candidate_count, default_feat_size + x, 1)
            elif xaxis == 'sparsity':
                feat_size = max(inv_densities)
                if yaxis == 'mat_time':
                    run = lambda: mat(candidate_count, feat_size, x)
                else:
                    m = mat(candidate_count, feat_size, x)
                
            if yaxis == 'query_time':
                repetition = 1000
                exeuctions = 100
                run = lambda : m.query()
                
            timings = timeit.Timer(run).repeat(repetition, exeuctions)
            # min timings are indicative of the hardware limitations
            series[mat_name].append(min(timings))
#             print mat_name, run().data
        
        
        plt.plot(real_xs, series[mat_name], label=mat_name)
    
    print 'Benchmark_' + xaxis + '_vs._' + yaxis + '\t' + '\t'.join([str(x) for x in real_xs])
    for mat_name, data in series.iteritems():
        print mat_name + '\t' + '\t'.join([str(x) for x in data])
#     show_plot(xaxis, yaxis)


if __name__ == '__main__':
    xaxis = ['feat_size', 'iterations', 'sparsity']
    yaxis = ['query_time','mat_time']
    for y in yaxis:
        for x in xaxis:
#             if x == 'feat_size' and y == 'mat_time': 
            run_benchmark(x, y)
    
    
