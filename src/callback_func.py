import sys

def audio_callback(arguments, mapping, queue):
    """
    fancy closure
    """
    def basic_callback(indata, frames, time, status, arguments=arguments, mapping=mapping, queue=queue):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        queue.put(indata[::arguments.downsample, mapping])

    return basic_callback


def audio_out_callback(arguments, mapping, queue):
    """
    fancy closure
    """
    def basic_callback(frames, time, status, arguments=arguments, mapping=mapping, queue=queue):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)

    return basic_callback