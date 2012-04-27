import copy

class SelectiveUnimplementer(object):
    def __init__(self, base, deletions):
        self._base = base
        self._deletions = deletions
    def unimplemented(self, *args, **kwargs):
        raise NotImplementedError
    def __getattr__(self, name):
        if name not in self._deletions:
            return getattr(self._base, name)
        else:
            return self.unimplemented
    def copy(self):
        return SelectiveUnimplementer(
            copy.deepcopy(self._base), self._deletions)
        
def make_null_toolchain(toolchain):
    """Creates a codepy toolchain which behaves identically to a given
    toolchain, but cannot actually compile code.  This allows us to
    check whether a binary has already been compiled, without paying
    the cost of compiling one if it has not been compiled."""
    
    return SelectiveUnimplementer(toolchain,
                                  set(["build_extension",
                                       "build_object",
                                       "link_extension"]))
