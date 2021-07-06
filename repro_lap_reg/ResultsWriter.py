import os
from pprint import pformat


class ResultsWriter(object):
    def __init__(self, fpath=None, delete_if_exists=True,
                 to_print=True, to_write=True, default_newlines=0):
        self.fpath = fpath
        self.delete_if_exists = delete_if_exists
        self.to_print = to_print
        self.to_write = to_write
        self.default_newlines = default_newlines

        if fpath is None:
            self.to_write = False
        else:
            if os.path.exists(self.fpath):
                if self.delete_if_exists:
                    os.remove(self.fpath)
                else:
                    raise ValueError('Warning {} already exists'.
                                     format(self.fpath))

    def write(self, text=None, newlines=None):

        if newlines is None:
            newlines = self.default_newlines

        # possibly format dicts to nicer strings
        if isinstance(text, dict):
            out = pformat(text)
        else:
            out = str(text)

        if self.to_print:

            if text is not None:
                print(out)
            else:
                print()

            if newlines:
                for _ in range(newlines):
                    print()

        if self.to_write:
            with open(self.fpath, "a") as log_file:
                if text is not None:
                    log_file.write(out)
                    log_file.write('\n')

                if newlines:
                    log_file.write('\n' * newlines)
