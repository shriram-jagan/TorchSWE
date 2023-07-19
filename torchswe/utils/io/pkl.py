import pickle as pkl
from torchswe.utils.data.states import States
from torchswe.utils.misc import DummyDict
from torchswe.utils.config import Config

def dump_solution(soln: States, runtime: DummyDict, config: Config, filename: str = None):
    """Dump mesh and solution variables to a pickle file"""

    if config.params.dump_solution:
        tidx = runtime.tidx

        d = {'w': soln.q[(0,)  + soln.domain.nonhalo_c],
             'hu': soln.q[(1,) + soln.domain.nonhalo_c],
             'hv': soln.q[(2,) + soln.domain.nonhalo_c],
             'h':  soln.p[(0,) + soln.domain.nonhalo_c],
             'u':  soln.p[(1,) + soln.domain.nonhalo_c],
             'v':  soln.p[(2,) + soln.domain.nonhalo_c],
             'x':  soln.domain.x.v,
             'y':  soln.domain.y.v,
             'dx': soln.domain.x.delta,
             'dy': soln.domain.y.delta,
             'time': float(runtime.cur_t),
             'iterations': float(runtime.counter),
            }

        # append '.pkl' if needed
        if filename is not None:
            split = filename.split('.')

            if len(split) > 1 and (split[-1] != 'pkl'):
                # e.g.: filename = 'x.extn'
                filename += '.pkl'
            elif len(split) == 1:
                # e.g.: filename = 'x'
                filename += '.pkl'
        else:
            filename = str(nplike.__name__) + '_allvars.pkl'

        pkl.dump(d, open(filename, "wb"))
