try:
    import matplotlib
except ImportError:
    collect_ignore = ["nilearn/plotting"]
