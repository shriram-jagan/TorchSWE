import numpy as np

# This script will print the shapes and sizes of the arrays
# that are used in TorchSWE (only some of the most 
# frequently used arrays in the time-stepping are included)


# INPUTS

# Ideally, this should match the discretization set in config.yaml
nx = 2000
ny = 500

# number of ghost points; don't change this
ngh = 2 

# size of double precision 
dtype_size = 8.0

d = {}

# States class
d['states.q'] = (3, ny+2*ngh,  nx+2*ngh)
d['states.p'] = (3, ny+2*ngh, nx+2*ngh)
d['states.s'] = (3, ny, nx)
d['states.ss'] = (3, ny, nx)
d['states.face.x.plus.q'] = (3, ny, nx+1)
d['states.face.x.plus.p'] = (3, ny, nx+1)
d['states.face.x.plus.a'] = (ny, nx+1)
d['states.face.x.plus.f'] = (3, ny, nx+1)

d['states.face.x.minus.q'] = (3, ny, nx+1)
d['states.face.x.minus.p'] = (3, ny, nx+1)
d['states.face.x.minus.a'] = (ny, nx+1)
d['states.face.x.minus.f'] = (3, ny, nx+1)

d['states.face.x.cf'] = (3, ny, nx+1)

d['states.face.y.plus.q'] = (3, ny+1, nx)
d['states.face.y.plus.p'] = (3, ny+1, nx)
d['states.face.y.plus.a'] = (ny+1, nx)
d['states.face.y.plus.f'] = (3, ny+1, nx)

d['states.face.y.minus.q'] = (3, ny+1, nx)
d['states.face.y.minus.p'] = (3, ny+1, nx)
d['states.face.y.minus.a'] = (ny+1, nx)
d['states.face.y.minus.f'] = (3, ny+1, nx)

d['states.face.y.cf'] = (3, ny+1, nx)
d['states.slpx'] = (3, ny, nx+2)
d['states.slpy'] = (3, ny+2, nx)

# Topo class
d['topo.v'] = (ny+ 2*ngh +1, nx+2*ngh+1)
d['topo.c'] = (ny+2*ngh, nx+2*ngh) 
d['topo.xf'] = (ny, nx+1)
d['topo.yf'] = (ny+1, nx)
d['topo.grad'] = (2, ny, nx)

print("(nx, ny): %-30s" % str((nx, ny)))
print()
print("%30s %20s %10s" %("Variable", "Shape", "Size (MB)"))
print('-'*65)

counter = 0
class_mem_mb = 0
total_mem_mb = 0
variable_in_class = list(d.keys())[0].split('.')[0] if len(d.keys())>0 else None
for variable, shape in d.items():
    size_mb = np.prod(shape)*dtype_size/1024**2
    class_mem_mb += size_mb
    total_mem_mb += size_mb

    # print blank line if necessary
    counter += 1
    if counter % 5 == 0 or variable.split('.')[0] != variable_in_class:
        print()

    if variable.split('.')[0] != variable_in_class:
        print("%30s %20s %10.3f" %("Memory used by class " + variable_in_class, " N/A", class_mem_mb))
        print()
        class_mem_mb = 0

    print("%30s %20s %10.3f" %(variable, str(shape), size_mb))

    # update the name of the class that this variable belongs to
    variable_in_class = variable.split('.')[0]

# pretty print
if counter % 5 != 0:
    print()

# Print total memory
print("%30s %20s %10.3f" %("Memory used by class " + variable_in_class, " N/A", class_mem_mb))
print()
print("%30s %20s %10.3f" %("Total memory used", " N/A", total_mem_mb))
