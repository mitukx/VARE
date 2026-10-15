"""Private POSIX grader owner. EOF on stdin kills this process's entire group.

The runner launches this leader in a new session and owns the only stdin
writer. Graders inherit neither that writer nor the private exit-status fd.
This contains ordinary descendants, not code that escapes its process group.
"""
import sys

# File execution adds src/vare to sys.path. Its types.py must not hide the
# standard-library types module when site/editable imports have not loaded it.
sys.path = sys.path[1:] if __name__ == '__main__' else sys.path

def main():
    import os
    import signal
    import struct
    import subprocess
    import threading

    status_fd = int(sys.argv[1])
    os.set_inheritable(status_fd, False)

    def watch_owner():
        try:
            while os.read(0, 4096):
                pass
        finally:
            # The guard remains the living group leader until this signal.
            # It never targets a persisted/reused PID or a different session.
            os.killpg(os.getpgrp(), signal.SIGKILL)

    watcher = threading.Thread(target=watch_owner, daemon=True)
    watcher.start()
    try:
        try:
            child = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, close_fds=True)
            code = child.wait()
        except OSError as exc:
            print('grader launch failed: ' + type(exc).__name__, file=sys.stderr, flush=True)
            code = 127
        os.write(status_fd, struct.pack('!i', code))
        os.close(status_fd)
        # EOF on the capture pipes still means exactly what it meant in the runner:
        # descendants retaining stdout/stderr must finish or hit the job timeout.
        os.close(1)
        os.close(2)
        watcher.join()
    finally:
        # A broken status pipe can race owner EOF. Never exit the leader on a
        # Python error before its watcher has stopped the remaining descendants.
        os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == '__main__':
    main()
