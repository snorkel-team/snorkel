import sys
sys.path.append('../') #adding Coral Home, temp hack
sys.path.append('../coral/')
sys.path.append('../coral/learning')
print 'printing....\n', sys.path
from coral.static_analysis.dependency_learning import find_dependencies

