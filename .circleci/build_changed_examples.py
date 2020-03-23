import os
import shlex
import subprocess

from pprint import pprint


# Assuming current working directory is nilearn/docs .
git_add_upstream_cmd = 'git remote add upstream https://github.com/nilearn/nilearn'
git_diff_examples_shell = subprocess.Popen(shlex.split(git_add_upstream_cmd),
                                           stderr=subprocess.STDOUT,
                                           stdout=subprocess.PIPE,
                                           )
output, errors = git_diff_examples_shell.communicate()


git_diff_examples_cmd = 'git diff upstream/master --name-only ../examples/**/*.py'
git_diff_examples_shell = subprocess.Popen(shlex.split(git_diff_examples_cmd),
                                           stderr=subprocess.STDOUT,
                                           stdout=subprocess.PIPE,
                                           )
output, errors = git_diff_examples_shell.communicate()
changed_examples = output.decode(encoding='utf8').split('\n')[:-1]
print(os.getcwd())
pprint(changed_examples)

# venv_activate_shell = subprocess.Popen(shlex.split("conda activate nilearn-py37-latest"), shell=True)
# print(venv_activate_shell.communicate())

sphinx_cmd = ('python -m sphinx -W -D sphinx_gallery_conf.filename_pattern={} '
              '-b html -d _build/doctrees . _build/html')
for changed_example_ in changed_examples:
    example_shell = subprocess.Popen(shlex.split(sphinx_cmd.format(changed_example_)), shell=True)
    print(example_shell.communicate())


